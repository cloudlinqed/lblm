//! blmrs dna — native port of the DNA-aware predictor `dna.py`, for FULL-GENOME scale.
//!
//! `dna.py` (design doc §49) reaches E. coli 1.908 / human chr21 1.616 bits/base by operating on
//! 2-bit bases with a codon (period-3 reading-frame) phase, base-history context orders, a verified
//! base-granular forward match, and a reverse-complement match for inverted repeats — but it is pure
//! Python (dict tables), so it cannot run whole chromosomes at speed. This is a faithful translation
//! using `strong.rs`-style bounded-RAM flat open-addressing tables (multiplicative hash + 8-bit
//! checksum tag) for the orders, so the same model runs at 10s-of-Mbase scale.
//!
//! The representation is the project's "induce the right unit" thesis made native: the unit is the
//! 2-bit base, the period is the codon (3), and inverted repeats are caught by reverse-complement —
//! none of `strong.rs`'s 8-bit byte assumptions. Usage: dna <path.2bit> [base_cap] [obits].

use std::env;
use std::fs;
use std::time::Instant;

const ORDERS: [usize; 8] = [1, 2, 3, 4, 6, 8, 12, 16]; // context orders in BASES
const NM: usize = 8;
const I_MT: usize = NM; // forward match
const I_RC: usize = NM + 1; // reverse-complement match
const NIN: usize = NM + 2;
const NW: usize = NIN + 1; // + bias
const MAXB: usize = 16;
const MULT: u64 = 0x9E37_79B9_7F4A_7C15;

const ALR: f64 = 0.01;
const DELTA: f64 = 0.5;
const RMS_DECAY: f64 = 0.999;
const RMS_EPS: f64 = 1e-3;

#[inline]
fn stretch(p: f64) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}
#[inline]
fn squash(t: f64) -> f64 {
    if t > 30.0 { 1.0 - 1e-6 } else if t < -30.0 { 1e-6 } else { 1.0 / (1.0 + (-t).exp()) }
}

/// Base-granular forward match (min length 20, hash-collision-verified) — ports dna.py `Match`.
struct Match {
    mask: u64,
    tab: Vec<u32>,
    minlen: usize,
    ptr: usize,
    len: usize,
    h: u64,
}
impl Match {
    fn new(hb: u32, minlen: usize) -> Self {
        Match { mask: (1u64 << hb) - 1, tab: vec![0u32; 1usize << hb], minlen, ptr: 0, len: 0, h: 0 }
    }
    #[inline]
    fn predict(&self, bases: &[u8], bp: usize, bit_in_base: usize, partial: u8) -> f64 {
        if self.len == 0 || self.ptr >= bp {
            return 0.0;
        }
        let pb = bases[self.ptr];
        let bit = if bit_in_base == 0 {
            (pb >> 1) & 1
        } else {
            if ((pb >> 1) & 1) != partial {
                return 0.0;
            }
            pb & 1
        };
        let st = 1.5 + 0.30 * (self.len.min(32) as f64);
        if bit == 1 { st } else { -st }
    }
    fn update_after_base(&mut self, bases: &[u8], bp: usize) {
        if self.len > 0 && self.ptr < bp {
            if bases[self.ptr] == bases[bp] {
                self.ptr += 1;
                self.len = (self.len + 1).min(65535);
            } else {
                self.len = 0;
                self.ptr = 0;
            }
        }
        self.h = ((self.h << 2) | bases[bp] as u64) & ((1u64 << 48) - 1); // last 24 bases
        if bp + 1 >= self.minlen {
            let hk = (self.h.wrapping_mul(2654435761) & self.mask) as usize;
            let prev = self.tab[hk] as usize;
            self.tab[hk] = (bp + 1) as u32;
            if self.len == 0 && prev >= self.minlen && prev <= bp {
                let mut ok = true; // VERIFY against hash collisions
                for j in 0..self.minlen {
                    if bases[prev - 1 - j] != bases[bp - j] {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    self.ptr = prev;
                    self.len = self.minlen;
                }
            }
        }
    }
}

/// Reverse-complement match (inverted repeats) — ports dna.py `RCMatch`. Shares the forward table.
struct RcMatch {
    k: usize,
    rcmask: u64,
    rc_h: u64,
    ptr: i64,
    len: usize,
}
impl RcMatch {
    fn new(k: usize) -> Self {
        RcMatch { k, rcmask: (1u64 << (2 * k)) - 1, rc_h: 0, ptr: -1, len: 0 }
    }
    #[inline]
    fn predict(&self, bases: &[u8], bp: usize, bit_in_base: usize, partial: u8) -> f64 {
        if self.len == 0 || self.ptr < 0 {
            return 0.0;
        }
        let pb = 3 - bases[self.ptr as usize]; // complement of the matched base
        let _ = bp;
        let bit = if bit_in_base == 0 {
            (pb >> 1) & 1
        } else {
            if ((pb >> 1) & 1) != partial {
                return 0.0;
            }
            pb & 1
        };
        let st = 1.5 + 0.30 * (self.len.min(32) as f64);
        if bit == 1 { st } else { -st }
    }
    fn update_after_base(&mut self, bases: &[u8], bp: usize, fwd_tab: &[u32], fwd_mask: u64) {
        if self.len > 0 && self.ptr >= 0 {
            if (3 - bases[self.ptr as usize]) == bases[bp] {
                self.ptr -= 1;
                self.len = (self.len + 1).min(65535);
                if self.ptr < 0 {
                    self.len = 0;
                }
            } else {
                self.len = 0;
                self.ptr = -1;
            }
        }
        self.rc_h = ((self.rc_h >> 2) | (((3 - bases[bp]) as u64) << (2 * (self.k - 1)))) & self.rcmask;
        if self.len == 0 && bp + 1 >= self.k {
            let p = fwd_tab[(self.rc_h.wrapping_mul(2654435761) & fwd_mask) as usize] as usize;
            if p >= self.k + 1 && p <= bp {
                let mut ok = true; // earlier forward k-mer = bases[p-K..p-1]
                for kk in 0..self.k {
                    if bases[p - self.k + kk] != 3 - bases[bp - kk] {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    self.ptr = (p - self.k) as i64 - 1; // continue backward, complemented
                    if self.ptr >= 0 {
                        self.len = self.k;
                    }
                }
            }
        }
    }
}

fn load_bases(path: &str, cap: usize) -> Vec<u8> {
    let raw = fs::read(path).expect("read input");
    let mut out: Vec<u8> = Vec::with_capacity(raw.len() * 4);
    for &b in &raw {
        out.push((b >> 6) & 3);
        out.push((b >> 4) & 3);
        out.push((b >> 2) & 3);
        out.push(b & 3);
        if cap > 0 && out.len() >= cap {
            out.truncate(cap);
            return out;
        }
    }
    out
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("data/ecoli.2bit");
    let cap: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let obits: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(23);
    let osize: usize = 1usize << obits;

    let bases = load_bases(path, cap);
    let n = bases.len();

    let masks: Vec<u64> = (0..=MAXB).map(|b| if 2 * b >= 64 { u64::MAX } else { (1u64 << (2 * b)) - 1 }).collect();
    let htmask: u64 = (1u64 << (2 * MAXB)) - 1;

    // flat bounded-RAM tables per order: 2 counts + 8-bit checksum tag
    let mut ocount: Vec<Vec<u32>> = (0..NM).map(|_| vec![0u32; 2 * osize]).collect();
    let mut otag: Vec<Vec<u8>> = (0..NM).map(|_| vec![0u8; osize]).collect();

    let mut w = [0.0f64; NW];
    let mut g = [0.0f64; NW];
    let mut mtch = Match::new(22, 20);
    let mut rc = RcMatch::new(24);

    let mut bhist: u64 = 0;
    let mut sts = [0.0f64; NW];
    let mut oslot = [0usize; NM];
    let mut tot = 0.0f64;
    let mut tail = 0.0f64;
    let mut tailn = 0usize;
    let split = (n as f64 * 0.8) as usize;
    let t0 = Instant::now();

    for bp in 0..n {
        let base = bases[bp];
        let codon = (bp % 3) as u64;
        let b0 = (base >> 1) & 1;
        let b1 = base & 1;
        for bit_in_base in 0..2usize {
            let ybit = if bit_in_base == 0 { b0 } else { b1 };
            let partial = if bit_in_base == 0 { 0u8 } else { b0 };
            let lobits = codon | ((bit_in_base as u64) << 2) | ((partial as u64) << 3);
            for k in 0..NM {
                let b = ORDERS[k];
                let key = ((bhist & masks[b]) << 4) | lobits;
                let h = key.wrapping_mul(MULT);
                let ti = (h >> (64 - obits)) as usize;
                let want = ((h >> (64 - obits - 8)) & 0xFF) as u8;
                if otag[k][ti] != want {
                    otag[k][ti] = want;
                    ocount[k][2 * ti] = 0;
                    ocount[k][2 * ti + 1] = 0;
                }
                oslot[k] = ti;
                let (n0, n1) = (ocount[k][2 * ti] as f64, ocount[k][2 * ti + 1] as f64);
                sts[k] = stretch((n1 + DELTA) / (n0 + n1 + 2.0 * DELTA));
            }
            sts[I_MT] = mtch.predict(&bases, bp, bit_in_base, partial);
            sts[I_RC] = rc.predict(&bases, bp, bit_in_base, partial);
            sts[NIN] = 1.0;

            let mut d = 0.0;
            for k in 0..NW {
                d += w[k] * sts[k];
            }
            let p = squash(d);
            let yf = ybit as f64;
            let cost = -(if ybit == 1 { p } else { 1.0 - p }).log2();
            tot += cost;
            if bp >= split {
                tail += cost;
                tailn += 1;
            }
            let err = yf - p;
            for k in 0..NW {
                let grad = err * sts[k];
                g[k] = RMS_DECAY * g[k] + (1.0 - RMS_DECAY) * grad * grad;
                w[k] += ALR * grad / (g[k].sqrt() + RMS_EPS);
            }
            let yi = ybit as usize;
            for k in 0..NM {
                ocount[k][2 * oslot[k] + yi] += 1;
            }
        }
        bhist = ((bhist << 2) | base as u64) & htmask;
        mtch.update_after_base(&bases, bp);
        rc.update_after_base(&bases, bp, &mtch.tab, mtch.mask);
    }

    let secs = t0.elapsed().as_secs_f64();
    let whole = tot / n as f64; // bits per base (2 bits summed per base)
    let last = if tailn > 0 { tail / (tailn as f64 / 2.0) } else { 0.0 };
    println!("path={}  bases={}  obits={}", path, n, obits);
    println!(
        "  blmrs-dna  whole = {:.4} bits/base   last-20% = {:.4}   (floor 2.0)   [{:.1}s, {:.2} Mbase/s]",
        whole, last, secs, (n as f64 / 1e6) / secs
    );
}

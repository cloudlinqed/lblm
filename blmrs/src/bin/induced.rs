//! blmrs induced — the period-DISCOVERING native engine (port of `real_scale.py`, + folded-in models).
//!
//! The bit-native "induce the representation" thesis, native and at scale. It DISCOVERS the predictive
//! unit/period `p` from the data (a quick held-out period scan), then runs unit-aligned online logistic
//! context mixing at `p` (integer rolling keys + `strong.rs`-style bounded-RAM flat tables + a 33-knot
//! APM), plus a universal **forward match** model (long-repeat predictor) and, when the discovered unit
//! is base-like (DNA, p even and small), a base-granular **reverse-complement** match (inverted repeats).
//! It is NOT told the unit: on text it discovers the byte (p=8); on 2-bit DNA the codon (p=6).
//!
//! Usage:  induced <path> [byte_cap] [obits] [mode]   mode = run | scan | test

use std::collections::HashMap;
use std::env;
use std::fs;
use std::time::Instant;

const MULT: u64 = 0x9E37_79B9_7F4A_7C15;
const C1: u64 = 0x2545_F491_4F6C_DD1D;
const DELTA: f64 = 0.2;
const LR: f64 = 0.02;
const WCOUNT: usize = 16;

#[inline]
fn stretch(p: f64) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}
#[inline]
fn squash(t: f64) -> f64 {
    if t > 30.0 { 1.0 - 1e-6 } else if t < -30.0 { 1e-6 } else { 1.0 / (1.0 + (-t).exp()) }
}

fn load_bits(path: &str, cap: usize) -> Vec<u8> {
    let mut raw = fs::read(path).expect("read input");
    if cap > 0 && raw.len() > cap {
        raw.truncate(cap);
    }
    let mut bits = Vec::with_capacity(raw.len() * 8);
    for &b in &raw {
        for j in (0..8).rev() {
            bits.push((b >> j) & 1);
        }
    }
    bits
}

/// 33-knot adaptive probability map (SSE) — ports `real_scale.py`'s APM.
struct Apm {
    t: Vec<f64>,
    ci: usize,
    cj: usize,
    cw: f64,
}
impl Apm {
    fn new(nc: usize) -> Self {
        let mut t = vec![0.0f64; nc * 33];
        for c in 0..nc {
            for j in 0..33 {
                t[c * 33 + j] = squash((j as f64 - 16.0) / 2.0);
            }
        }
        Apm { t, ci: 0, cj: 0, cw: 0.0 }
    }
    #[inline]
    fn refine(&mut self, pr: f64, cx: usize) -> f64 {
        let mut v = stretch(pr) * 2.0 + 16.0;
        if v < 0.0 { v = 0.0; } else if v > 31.999 { v = 31.999; }
        let i = v as usize;
        let w = v - i as f64;
        self.ci = cx; self.cj = i; self.cw = w;
        let b = cx * 33;
        self.t[b + i] * (1.0 - w) + self.t[b + i + 1] * w
    }
    #[inline]
    fn learn(&mut self, y: f64, rate: f64) {
        let b = self.ci * 33;
        let (j, w) = (self.cj, self.cw);
        self.t[b + j] += rate * (1.0 - w) * (y - self.t[b + j]);
        self.t[b + j + 1] += rate * w * (y - self.t[b + j + 1]);
    }
}

/// Forward match at the DISCOVERED unit — symbol-granular (byte for text, codon for DNA), unit-aligned
/// like `strong.rs`. Predicts the bit at the current phase from the matched symbol; collision-verified.
struct SymMatch {
    mask: u64,
    tab: Vec<u32>,
    minlen: usize,
    ptr: usize,
    len: usize,
    h: u64,
    p: usize,
}
impl SymMatch {
    fn new(hb: u32, minlen: usize, p: usize) -> Self {
        SymMatch { mask: (1u64 << hb) - 1, tab: vec![0u32; 1usize << hb], minlen, ptr: 0, len: 0, h: 0, p }
    }
    #[inline]
    fn predict(&self, syms: &[u32], phase: usize, sym_pos: usize) -> f64 {
        if self.len == 0 || self.ptr >= sym_pos {
            return 0.0;
        }
        let pb = (syms[self.ptr] >> (self.p - 1 - phase)) & 1;
        let st = 1.6 + 0.35 * (self.len.min(28) as f64);
        if pb == 1 { st } else { -st }
    }
    fn update_after_sym(&mut self, syms: &[u32], sym_pos: usize) {
        if self.len > 0 && self.ptr < sym_pos {
            if syms[self.ptr] == syms[sym_pos] {
                self.ptr += 1;
                self.len = (self.len + 1).min(65535);
            } else {
                self.len = 0;
                self.ptr = 0;
            }
        }
        let shift = self.p.min(16);
        self.h = ((self.h << shift) | syms[sym_pos] as u64) & ((1u64 << 48) - 1);
        if sym_pos + 1 >= self.minlen {
            let hk = (self.h.wrapping_mul(2654435761) & self.mask) as usize;
            let prev = self.tab[hk] as usize;
            self.tab[hk] = (sym_pos + 1) as u32;
            if self.len == 0 && prev >= self.minlen && prev <= sym_pos {
                let mut ok = true;
                for j in 0..self.minlen {
                    if syms[prev - 1 - j] != syms[sym_pos - j] {
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

/// Reverse-complement match (DNA inverted repeats) — base-granular, shares the forward base table.
/// Ports `dna.rs`/`dna.py` RCMatch. Active only when the discovered unit is base-like (DNA mode).
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
    fn predict(&self, syms: &[u32], sym_pos: usize, phase: usize) -> f64 {
        if self.len == 0 || self.ptr < 0 || self.ptr as usize >= sym_pos {
            return 0.0;
        }
        let pb = 3 - syms[self.ptr as usize]; // complement base
        let bit = (pb >> (1 - phase)) & 1;
        let st = 1.5 + 0.30 * (self.len.min(32) as f64);
        if bit == 1 { st } else { -st }
    }
    fn update_after_base(&mut self, syms: &[u32], bp: usize, fwd_tab: &[u32], fwd_mask: u64) {
        if self.len > 0 && self.ptr >= 0 {
            if (3 - syms[self.ptr as usize]) == syms[bp] {
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
        self.rc_h = ((self.rc_h >> 2) | (((3 - syms[bp]) as u64) << (2 * (self.k - 1)))) & self.rcmask;
        if self.len == 0 && bp + 1 >= self.k {
            let p = fwd_tab[(self.rc_h.wrapping_mul(2654435761) & fwd_mask) as usize] as usize;
            if p >= self.k + 1 && p <= bp {
                let mut ok = true;
                for kk in 0..self.k {
                    if syms[p - self.k + kk] != 3 - syms[bp - kk] {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    self.ptr = (p - self.k) as i64 - 1;
                    if self.ptr >= 0 {
                        self.len = self.k;
                    }
                }
            }
        }
    }
}

fn period_score(bits: &[u8], split: usize, end: usize, p: usize) -> f64 {
    let mut tab: HashMap<u64, [u32; 2]> = HashMap::new();
    for i in 0..split {
        let ph = i % p;
        let mut partial = 0u64;
        for k in (i - ph)..i {
            partial = (partial << 1) | bits[k] as u64;
        }
        let key = ((ph as u64) << 40) | partial;
        let c = tab.entry(key).or_insert([0, 0]);
        c[bits[i] as usize] += 1;
    }
    let ones: u64 = bits[..split].iter().map(|&b| b as u64).sum();
    let g1 = (ones as f64 + 0.5) / (split as f64 + 1.0);
    let mut tot = 0.0;
    for i in split..end {
        let ph = i % p;
        let mut partial = 0u64;
        for k in (i - ph)..i {
            partial = (partial << 1) | bits[k] as u64;
        }
        let key = ((ph as u64) << 40) | partial;
        let p1 = match tab.get(&key) {
            None => g1,
            Some(c) => (c[1] as f64 + 0.5) / (c[0] as f64 + c[1] as f64 + 1.0),
        };
        let pp = if bits[i] == 1 { p1 } else { 1.0 - p1 };
        tot += -(pp.max(1e-12)).log2();
    }
    tot / (end - split) as f64
}

fn discover_period(bits: &[u8], scan_cap: usize) -> (usize, Vec<(usize, f64)>) {
    // Multi-window vote: scan several windows spread across the data, SKIP low-signal (near-constant)
    // windows — e.g. a telomere/low-complexity prefix — and take the most-voted best period. Robust to
    // a non-representative start. Falls back to a single prefix scan for small inputs.
    let total = bits.len();
    let win = scan_cap.min(total);
    let split = (win as f64 * 0.8) as usize;
    let scan_at = |start: usize| -> Vec<(usize, f64)> {
        let seg = &bits[start..start + win];
        (2..=12usize).map(|p| (p, period_score(seg, split, win, p))).collect()
    };
    // Use the FIRST REPRESENTATIVE window: scan from the start forward and take the first window that
    // is not low-complexity. A telomere / N-run scores near 0 (trivially compressible); real text ~0.58,
    // real DNA ~0.97. Skipping low-score windows steps past a non-representative prefix (e.g. chr21).
    let n_win = if total > win * 6 { 6 } else { 1 };
    let mut fallback = scan_at(0);
    for fr in 0..n_win {
        let start = ((total.saturating_sub(win)) * fr / n_win.max(1)) & !1;
        let s = scan_at(start);
        let mn = s.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
        if fr == 0 {
            fallback = s.clone();
        }
        if mn < 0.30 {
            continue; // low-complexity window — skip past it
        }
        let bp = s.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
        return (bp, s);
    }
    let bp = fallback.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
    (bp, fallback)
}

/// The induced engine at a fixed period p. Returns (whole, last-20%, cost-over-first-`cost_chk`-bits).
/// `cost_chk` accumulates cost only for i < cost_chk while still PROCESSING the whole stream — the
/// causality probe (a look-ahead bug would make a future-bit flip change this prefix cost).
fn run(bits: &[u8], p: usize, maxb_req: usize, obits: u32, sse: bool, cost_chk: usize) -> (f64, f64, f64) {
    let n = bits.len();
    let osize = 1usize << obits;
    let maxb = maxb_req.min((63usize.saturating_sub(p)) / p);
    let unit_k = maxb + 1;
    let ci = unit_k; // counter
    let mi = unit_k + 1; // forward match
    let ri = unit_k + 2; // reverse-complement match (DNA mode; else always 0)
    let bias = unit_k + 3;
    let k_total = unit_k + 4;

    let mut ocount: Vec<Vec<u32>> = (0..unit_k).map(|_| vec![0u32; 2 * osize]).collect();
    let mut otag: Vec<Vec<u8>> = (0..unit_k).map(|_| vec![0u8; osize]).collect();
    let mut ccount = vec![0u32; 2 * p * (WCOUNT + 1)];
    let mut w = vec![0.0f64; k_total];
    let mut sts = vec![0.0f64; k_total];
    let mut oslot = vec![0usize; unit_k];
    let mut apm = if sse { Some(Apm::new(256)) } else { None };
    // match granularity follows the discovered unit: base (2-bit) for DNA, byte/unit for text.
    let dna_mode = p % 2 == 0 && p <= 6;
    let mu = if dna_mode { 2 } else { p };
    let mut mtch = SymMatch::new(22, if dna_mode { 20 } else { 4 }, mu);
    let mut rc = if dna_mode { Some(RcMatch::new(24)) } else { None };
    let mut syms: Vec<u32> = Vec::with_capacity(n / mu + 1);
    let mut cur_sym: u32 = 0;

    let wbits = (maxb * p + p).max(WCOUNT) + 1;
    let histmask: u64 = if wbits >= 64 { u64::MAX } else { (1u64 << wbits) - 1 };
    let countmask: u64 = (1u64 << WCOUNT) - 1;
    let mut hist: u64 = 0;

    let split = (n as f64 * 0.8) as usize;
    let mut tot = 0.0;
    let mut tail = 0.0;
    let mut tailn = 0usize;
    let mut tot_chk = 0.0;

    for i in 0..n {
        let ph = i % p;
        let mut d = 0.0;
        for k in 0..unit_k {
            let len = ph + k * p;
            let mask = if len >= 64 { u64::MAX } else { (1u64 << len) - 1 };
            let h = (hist & mask).wrapping_mul(MULT) ^ (ph as u64).wrapping_mul(C1) ^ (k as u64).wrapping_mul(0x9E37_79B1);
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
            d += w[k] * sts[k];
        }
        let pc = (hist & countmask).count_ones() as usize;
        let cslot = 2 * (ph * (WCOUNT + 1) + pc);
        let (n0, n1) = (ccount[cslot] as f64, ccount[cslot + 1] as f64);
        sts[ci] = stretch((n1 + DELTA) / (n0 + n1 + 2.0 * DELTA));
        d += w[ci] * sts[ci];
        sts[mi] = mtch.predict(&syms, i % mu, syms.len());
        d += w[mi] * sts[mi];
        sts[ri] = rc.as_ref().map_or(0.0, |r| r.predict(&syms, syms.len(), i % mu));
        d += w[ri] * sts[ri];
        sts[bias] = 1.0;
        d += w[bias];

        let pmix = squash(d);
        let pr = if let Some(a) = apm.as_mut() {
            let p2 = a.refine(pmix, (hist & 0xff) as usize);
            ((pmix + 3.0 * p2) * 0.25).clamp(1e-6, 1.0 - 1e-6)
        } else {
            pmix
        };

        let y = bits[i];
        let cost = -(if y == 1 { pr } else { 1.0 - pr }).log2();
        tot += cost;
        if i < cost_chk {
            tot_chk += cost;
        }
        if i >= split {
            tail += cost;
            tailn += 1;
        }
        if let Some(a) = apm.as_mut() {
            a.learn(y as f64, 0.04);
        }
        let err = y as f64 - pmix;
        for k in 0..k_total {
            w[k] += LR * err * sts[k];
        }
        let yi = y as usize;
        for k in 0..unit_k {
            ocount[k][2 * oslot[k] + yi] += 1;
        }
        ccount[cslot + yi] += 1;
        cur_sym = (cur_sym << 1) | y as u32;
        if i % mu == mu - 1 {
            syms.push(cur_sym);
            let sp = syms.len() - 1;
            mtch.update_after_sym(&syms, sp);
            if let Some(r) = rc.as_mut() {
                r.update_after_base(&syms, sp, &mtch.tab, mtch.mask);
            }
            cur_sym = 0;
        }
        hist = ((hist << 1) | y as u64) & histmask;
    }
    let whole = tot / n as f64;
    let last = if tailn > 0 { tail / tailn as f64 } else { 0.0 };
    (whole, last, tot_chk)
}

/// Causality: process both streams in FULL; flipping a bit AFTER the checkpoint must not change the
/// cost over earlier bits (a look-ahead bug would).
fn causality_ok(bits: &[u8], p: usize, obits: u32) -> bool {
    let f = bits.len() / 2;
    let (_, _, base) = run(bits, p, 6, obits, true, f);
    let mut flipped = bits.to_vec();
    let g = (f + 50).min(bits.len() - 1);
    flipped[g] ^= 1;
    let (_, _, after) = run(&flipped, p, 6, obits, true, f);
    (base - after).abs() < 1e-9
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("data/corpus.txt");
    let cap: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let obits: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(22);
    let mode = args.get(4).map(|s| s.as_str()).unwrap_or("run");

    let bits = load_bits(path, cap);
    if bits.is_empty() {
        println!("empty input");
        return;
    }
    // a SMALL scan window favours true periodicity over context-length; 80k bits finds byte=8 / codon=6.
    let (p, scan) = discover_period(&bits, 80_000);

    if mode == "scan" {
        print!("path={}  bits={}  period scan:", path, bits.len());
        for (q, s) in &scan {
            print!("  p{}:{:.3}", q, s);
        }
        println!("\n  -> discovered unit period p = {}", p);
        return;
    }
    if mode == "test" {
        let d1 = run(&bits, p, 6, obits, true, bits.len());
        let d2 = run(&bits, p, 6, obits, true, bits.len());
        println!("path={}  bits={}  discovered p={}", path, bits.len(), p);
        println!("  determinism : {}  ({:.6} == {:.6})", d1 == d2, d1.0, d2.0);
        println!("  causality   : {}  (future-bit flip leaves prefix cost identical)", causality_ok(&bits, p, obits));
        return;
    }

    let t0 = Instant::now();
    let (whole, last, _) = run(&bits, p, 8, obits, true, bits.len());
    let secs = t0.elapsed().as_secs_f64();
    print!("path={}  bits={}  period scan:", path, bits.len());
    for (q, s) in &scan {
        print!("  p{}:{:.3}", q, s);
    }
    println!();
    println!(
        "  blmrs-induced  discovered p={}  whole={:.4}  last-20%={:.4}  bits/bit   [{:.1}s, {:.2} Mbits/s]",
        p, whole, last, secs, (bits.len() as f64 / 1e6) / secs
    );
}

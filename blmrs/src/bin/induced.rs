//! blmrs induced — the period-DISCOVERING native engine (port of `real_scale.py`).
//!
//! The bit-native "induce the representation" thesis, native and at scale. It DISCOVERS the predictive
//! unit/period `p` from the data (a quick held-out period scan), then runs unit-aligned online logistic
//! context mixing at `p` (integer rolling keys + `strong.rs`-style bounded-RAM flat tables + an SSE
//! stage). It is NOT told the unit: on English text it discovers the byte (p=8); on 2-bit-packed DNA it
//! discovers the codon (p=6). Faithful to `real_scale.py` (KT counts, logit mixing, SGD weights, the
//! 33-knot APM); causal (only bits strictly before i predict bit i).
//!
//! Usage:  induced <path> [byte_cap] [obits] [mode]
//!   mode = run (default) | scan (period scan only) | test (causality + determinism self-tests)

use std::collections::HashMap;
use std::env;
use std::fs;
use std::time::Instant;

const MULT: u64 = 0x9E37_79B9_7F4A_7C15;
const C1: u64 = 0x2545_F491_4F6C_DD1D;
const DELTA: f64 = 0.2;
const LR: f64 = 0.02;
const WCOUNT: usize = 16; // popcount window (bits) for the running-counter context

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

/// 33-knot adaptive probability map (SSE) — ports `real_scale.py`'s APM exactly.
struct Apm {
    t: Vec<f64>, // nc * 33
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
        if v < 0.0 {
            v = 0.0;
        } else if v > 31.999 {
            v = 31.999;
        }
        let i = v as usize;
        let w = v - i as f64;
        self.ci = cx;
        self.cj = i;
        self.cw = w;
        let base = cx * 33;
        self.t[base + i] * (1.0 - w) + self.t[base + i + 1] * w
    }
    #[inline]
    fn learn(&mut self, y: f64, rate: f64) {
        let base = self.ci * 33;
        let (j, w) = (self.cj, self.cw);
        self.t[base + j] += rate * (1.0 - w) * (y - self.t[base + j]);
        self.t[base + j + 1] += rate * w * (y - self.t[base + j + 1]);
    }
}

/// Quick held-out period score: key = (phase, partial-bits-since-phase-0); lower bits/bit is better.
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
    let end = scan_cap.min(bits.len());
    let split = (end as f64 * 0.8) as usize;
    let mut best_p = 8usize;
    let mut best = f64::INFINITY;
    let mut scan = Vec::new();
    for p in 2..=12usize {
        let s = period_score(bits, split, end, p);
        scan.push((p, s));
        if s < best {
            best = s;
            best_p = p;
        }
    }
    (best_p, scan)
}

/// The induced engine at a fixed period p. Returns (whole, last-20%) bits/bit.
fn run(bits: &[u8], p: usize, maxb_req: usize, obits: u32, sse: bool) -> (f64, f64) {
    let n = bits.len();
    let osize = 1usize << obits;
    // cap orders so the per-order window (phase + B*p) fits in u64
    let maxb = maxb_req.min((63usize.saturating_sub(p)) / p);
    let unit_k = maxb + 1;
    let k_total = unit_k + 1 + 1; // unit orders + counter + bias
    let ci = unit_k; // counter input index
    let bias = unit_k + 1;

    let mut ocount: Vec<Vec<u32>> = (0..unit_k).map(|_| vec![0u32; 2 * osize]).collect();
    let mut otag: Vec<Vec<u8>> = (0..unit_k).map(|_| vec![0u8; osize]).collect();
    let cnt_n = p * (WCOUNT + 1);
    let mut ccount = vec![0u32; 2 * cnt_n];

    let mut w = vec![0.0f64; k_total];
    let mut sts = vec![0.0f64; k_total];
    let mut oslot = vec![0usize; unit_k];
    let mut apm = if sse { Some(Apm::new(256)) } else { None };

    let wbits = (maxb * p + p).max(WCOUNT) + 1;
    let histmask: u64 = if wbits >= 64 { u64::MAX } else { (1u64 << wbits) - 1 };
    let countmask: u64 = (1u64 << WCOUNT) - 1;
    let mut hist: u64 = 0;

    let split = (n as f64 * 0.8) as usize;
    let mut tot = 0.0;
    let mut tail = 0.0;
    let mut tailn = 0usize;

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
            let st = stretch((n1 + DELTA) / (n0 + n1 + 2.0 * DELTA));
            sts[k] = st;
            d += w[k] * st;
        }
        // running-counter context (Path B aggregation)
        let pc = (hist & countmask).count_ones() as usize;
        let cslot = 2 * (ph * (WCOUNT + 1) + pc);
        let (n0, n1) = (ccount[cslot] as f64, ccount[cslot + 1] as f64);
        let st = stretch((n1 + DELTA) / (n0 + n1 + 2.0 * DELTA));
        sts[ci] = st;
        d += w[ci] * st;
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
        let yf = y as f64;
        let cost = -(if y == 1 { pr } else { 1.0 - pr }).log2();
        tot += cost;
        if i >= split {
            tail += cost;
            tailn += 1;
        }
        if let Some(a) = apm.as_mut() {
            a.learn(yf, 0.04);
        }
        let err = yf - pmix;
        for k in 0..k_total {
            w[k] += LR * err * sts[k];
        }
        let yi = y as usize;
        for k in 0..unit_k {
            ocount[k][2 * oslot[k] + yi] += 1;
        }
        ccount[cslot + yi] += 1;
        hist = ((hist << 1) | y as u64) & histmask;
    }
    let whole = tot / n as f64;
    let last = if tailn > 0 { tail / tailn as f64 } else { 0.0 };
    (whole, last)
}

/// Cumulative cost over the first `upto` bits while processing the WHOLE stream (so a look-ahead bug
/// — reading any bit at position >= the one being predicted — would change the result).
fn prefix_cost(bits: &[u8], p: usize, obits: u32, upto: usize) -> f64 {
    let n = bits.len();
    let osize = 1usize << obits;
    let maxb = 6usize.min((63usize.saturating_sub(p)) / p);
    let unit_k = maxb + 1;
    let k_total = unit_k + 2;
    let ci = unit_k;
    let bias = unit_k + 1;
    let mut ocount: Vec<Vec<u32>> = (0..unit_k).map(|_| vec![0u32; 2 * osize]).collect();
    let mut otag: Vec<Vec<u8>> = (0..unit_k).map(|_| vec![0u8; osize]).collect();
    let mut ccount = vec![0u32; 2 * p * (WCOUNT + 1)];
    let mut w = vec![0.0f64; k_total];
    let mut sts = vec![0.0f64; k_total];
    let mut oslot = vec![0usize; unit_k];
    let mut apm = Apm::new(256);
    let wbits = (maxb * p + p).max(WCOUNT) + 1;
    let histmask: u64 = if wbits >= 64 { u64::MAX } else { (1u64 << wbits) - 1 };
    let countmask: u64 = (1u64 << WCOUNT) - 1;
    let mut hist: u64 = 0;
    let mut tot = 0.0;
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
        sts[bias] = 1.0;
        d += w[bias];
        let pmix = squash(d);
        let p2 = apm.refine(pmix, (hist & 0xff) as usize);
        let pr = ((pmix + 3.0 * p2) * 0.25).clamp(1e-6, 1.0 - 1e-6);
        let y = bits[i];
        if i < upto {
            tot += -(if y == 1 { pr } else { 1.0 - pr }).log2();
        }
        apm.learn(y as f64, 0.04);
        let err = y as f64 - pmix;
        for k in 0..k_total {
            w[k] += LR * err * sts[k];
        }
        for k in 0..unit_k {
            ocount[k][2 * oslot[k] + y as usize] += 1;
        }
        ccount[cslot + y as usize] += 1;
        hist = ((hist << 1) | y as u64) & histmask;
    }
    tot
}

/// Causality: flipping a FUTURE bit must not change the cost over earlier bits. Both streams are
/// processed in FULL; a look-ahead bug would make the future flip perturb the prefix cost.
fn causality_ok(bits: &[u8], p: usize, obits: u32) -> bool {
    let f = bits.len() / 2;
    let base = prefix_cost(bits, p, obits, f);
    let mut flipped = bits.to_vec();
    let g = (f + 50).min(bits.len() - 1);
    flipped[g] ^= 1; // strictly AFTER the checkpoint f
    let after = prefix_cost(&flipped, p, obits, f);
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
    // a SMALL scan window favours true periodicity over context-length (long p wins on a big window
    // just by conditioning on more bits); 80k bits matches real_scale.py and finds byte=8 / codon=6.
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
        let det1 = run(&bits, p, 6, obits, true);
        let det2 = run(&bits, p, 6, obits, true);
        println!("path={}  bits={}  discovered p={}", path, bits.len(), p);
        println!("  determinism : {}  ({:.6} == {:.6})", det1 == det2, det1.0, det2.0);
        println!("  causality   : {}  (future-bit flip leaves prefix cost identical)", causality_ok(&bits, p, obits));
        return;
    }

    let t0 = Instant::now();
    let (whole, last) = run(&bits, p, 8, obits, true);
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

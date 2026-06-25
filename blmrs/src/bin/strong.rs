//! blmrs strong — the strong bit-native context-mixing predictor (extends the `mixnsfast.py` port).
//!
//! Models: byte-orders 0..7, hashed high orders {8,12,16,24,32}, 3 sparse models (non-adjacent byte
//! pairs), a word model + a previous-word model (text bigrams), and two byte-match models (min len
//! 5 and 8). These feed TWO context-selected logistic mixers (selected by order-1 and by order-2)
//! plus a global mixer, combined by a 4-weight final mixer, then a chain of 6 SSE/APM stages
//! (prev-byte, order-2, partial-byte, match-length, word-hash, order-3 contexts). Non-stationary
//! count recency-halving + RMSProp adaptive learning. Orders/sparse/word use flat fixed-size
//! open-addressing tables (high-bit multiplicative hash + 8-bit checksum tags) -> bounded memory;
//! the high orders use merged hashed tables.
//!
//! History: started as a faithful port of `mixnsfast.py` (verified bit-equal at small scale), then
//! improved — order-7, a 2nd (order-2) mixer, +2 SSE stages, +2 sparse models, the previous-word
//! model, and tuned DELTA/learning-rates — for ~2.5% lower bits/bit on real text (measured 11 MB,
//! see learned_binary_address_machine.md §63). Hyperparameters DELTA/ALR/ALRF are env-overridable.
//! Usage: strong <path> [byte_cap] [obits].

use std::env;
use std::fs;
use std::time::Instant;

const ORDERS: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
const HORDERS: [usize; 5] = [8, 12, 16, 24, 32];
const NH: usize = 5;
const NM: usize = 8;
const NSP: usize = 3; // sparse models over non-adjacent byte pairs (offsets below)
const SPOFF: [(usize, usize); NSP] = [(2, 3), (1, 4), (3, 6)];
const NIN: usize = NM + NH + NSP + 4; // [orders][hi orders][sparse..][word][word2][match][match2] then bias
const NW: usize = NIN + 1; // mixer weights (inputs + bias)
const NSEL: usize = 8 * 256;
const HBITS: u32 = 22;
const CLIMIT: u32 = 255;
const MAXB: usize = 7;
const MULT: u64 = 0x9E37_79B9_7F4A_7C15;

const RMS_DECAY: f64 = 0.9999;
const RMS_EPS: f64 = 1e-4;

#[inline]
fn envf(name: &str, default: f64) -> f64 {
    env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

#[inline]
fn stretch(p: f64) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}
#[inline]
fn squash(t: f64) -> f64 {
    if t > 30.0 { 1.0 - 1e-6 } else if t < -30.0 { 1e-6 } else { 1.0 / (1.0 + (-t).exp()) }
}

/// Adaptive Probability Map (SSE) — flat knot table over the stretch domain, per context.
struct Apm {
    k: usize,
    smax: f64,
    rate: f64,
    step: f64,
    t: Vec<f64>, // n_ctx * k
    lo: usize,
    w: f64,
    c: usize,
}
impl Apm {
    fn new(n_ctx: usize, k: usize, rate: f64) -> Self {
        let smax = 8.0;
        let step = 2.0 * smax / (k as f64 - 1.0);
        let mut t = vec![0.0f64; n_ctx * k];
        for j in 0..k {
            let v = squash(-smax + j as f64 * step);
            for c in 0..n_ctx {
                t[c * k + j] = v;
            }
        }
        Apm { k, smax, rate, step, t, lo: 0, w: 0.0, c: 0 }
    }
    #[inline]
    fn refine(&mut self, p: f64, cx: usize) -> f64 {
        let s = stretch(p);
        let (lo, w);
        if s <= -self.smax {
            lo = 0; w = 0.0;
        } else if s >= self.smax {
            lo = self.k - 2; w = 1.0;
        } else {
            let x = (s + self.smax) / self.step;
            let mut l = x as usize;
            if l >= self.k - 1 { l = self.k - 2; }
            lo = l; w = x - l as f64;
        }
        self.lo = lo; self.w = w; self.c = cx;
        let base = cx * self.k;
        self.t[base + lo] * (1.0 - w) + self.t[base + lo + 1] * w
    }
    #[inline]
    fn update(&mut self, y: f64) {
        let base = self.c * self.k;
        let (lo, w, rt) = (self.lo, self.w, self.rate);
        self.t[base + lo] += rt * (1.0 - w) * (y - self.t[base + lo]);
        self.t[base + lo + 1] += rt * w * (y - self.t[base + lo + 1]);
    }
}

struct MatchModel {
    mask: u64,
    tab: Vec<u32>,
    minlen: usize,
    ptr: usize,
    len: usize,
    h: u64,
}
impl MatchModel {
    fn new(hash_bits: u32, minlen: usize) -> Self {
        MatchModel { mask: (1u64 << hash_bits) - 1, tab: vec![0u32; 1usize << hash_bits], minlen, ptr: 0, len: 0, h: 0 }
    }
    #[inline]
    fn predicted(&self, hist: &[u8], phase: usize, byte_pos: usize) -> f64 {
        if self.len == 0 || self.ptr >= byte_pos {
            return 0.0;
        }
        let pb = (hist[self.ptr] >> (7 - phase)) & 1;
        let st = 1.6 + 0.35 * (self.len.min(28) as f64);
        if pb == 1 { st } else { -st }
    }
    fn update_after_byte(&mut self, hist: &[u8], byte_pos: usize) {
        if self.len > 0 && self.ptr < byte_pos {
            if hist[self.ptr] == hist[byte_pos] {
                self.ptr += 1; self.len = (self.len + 1).min(65535);
            } else {
                self.len = 0; self.ptr = 0;
            }
        }
        let b = hist[byte_pos] as u64;
        self.h = ((self.h << 8) | b) & 0xFFFF_FFFF_FFFF;
        if byte_pos + 1 >= self.minlen {
            let hk = (self.h.wrapping_mul(2654435761) & self.mask) as usize;
            let prev = self.tab[hk] as usize;
            self.tab[hk] = (byte_pos + 1) as u32;
            if self.len == 0 && prev != 0 && prev <= byte_pos {
                self.ptr = prev; self.len = self.minlen;
            }
        }
    }
}

#[allow(unused_assignments)] // sp_slot/wd_slot are seeded then reassigned every iteration
fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("data/corpus.txt");
    let cap: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300_000);
    let obits: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(23);
    let osize: usize = 1usize << obits;
    // env-tunable hyperparameters (defaults = the strong baseline) for fast sweeps without recompiling
    let delta = envf("DELTA", 0.08);     // tuned (was 0.18): lower smoothing -> sharper high-count contexts
    let d2 = 2.0 * delta;
    let alr_lr = envf("ALR", 0.0010);    // tuned mixer LR (was 0.0013)
    let alr_lrf = envf("ALRF", 0.0010);  // tuned final-mixer LR

    let mut raw = fs::read(path).expect("read input");
    if cap > 0 && raw.len() > cap { raw.truncate(cap); }
    let n = raw.len() * 8;
    let mut bits: Vec<u8> = Vec::with_capacity(n);
    for &byte in &raw {
        for j in (0..8).rev() { bits.push((byte >> j) & 1); }
    }

    let maskb: Vec<u64> = (0..=MAXB).map(|l| if 8 * l >= 64 { u64::MAX } else { (1u64 << (8 * l)) - 1 }).collect();
    let lr_final = 0.01f64;
    let hmask: u64 = (1u64 << HBITS) - 1;

    // flat tables (orders / sparse / word): counts (2 per slot) + 8-bit checksum tags
    let mut ocount: Vec<Vec<u32>> = (0..NM).map(|_| vec![0u32; 2 * osize]).collect();
    let mut otag: Vec<Vec<u8>> = (0..NM).map(|_| vec![0u8; osize]).collect();
    let mut spc: Vec<Vec<u32>> = (0..NSP).map(|_| vec![0u32; 2 * osize]).collect();
    let mut sptag: Vec<Vec<u8>> = (0..NSP).map(|_| vec![0u8; osize]).collect();
    let mut wdc = vec![0u32; 2 * osize]; let mut wdtag = vec![0u8; osize];
    let mut wdc2 = vec![0u32; 2 * osize]; let mut wdtag2 = vec![0u8; osize]; // previous-word model
    // high orders: merged hashed tables (no tags), exactly like the Python
    let mut htab: Vec<Vec<u32>> = (0..NH).map(|_| vec![0u32; 2 * (1usize << HBITS)]).collect();

    let mut mixers = vec![[0.0f64; NW]; NSEL];
    let mut mixers_g = vec![[0.0f64; NW]; NSEL];
    // a SECOND context-selected mixer, partitioned by order-2 (prev_byte,prev2) instead of order-1
    const NSEL2: usize = 8 * 2048;
    let mut mixers2 = vec![[0.0f64; NW]; NSEL2];
    let mut mixers2_g = vec![[0.0f64; NW]; NSEL2];
    let mut gmix = [0.0f64; NW];
    let mut gmix_g = [0.0f64; NW];
    let mut final_w = [0.3f64, 0.3, 0.2, 0.0]; // [w_sel, w_global, w_sel2, bias]
    let mut final_g = [0.0f64; 4];
    let mut apm1 = Apm::new(256 * 8, 33, 0.007);
    let mut apm2 = Apm::new(1024, 33, 0.005);
    let mut apm3 = Apm::new(256 * 8, 33, 0.006); // ctx = current partial byte + phase
    let mut apm4 = Apm::new(256, 33, 0.005); // ctx = match-length bucket + phase
    let mut apm5 = Apm::new(4096, 33, 0.005); // ctx = word hash + phase (text structure)
    let mut apm6 = Apm::new(4096, 33, 0.005); // ctx = order-3 byte hash + phase
    let mut mm = MatchModel::new(22, 5);
    let mut mm2 = MatchModel::new(22, 8); // a second, longer match model (min length 8)

    let mut hist: Vec<u8> = Vec::with_capacity(raw.len());
    let mut cur: u64 = 0;
    let mut phase: usize = 0;
    let mut prev_byte: u64 = 0;
    let mut prev2: u64 = 0;
    let mut word_hash: u64 = 0;
    let mut byte_pos: usize = 0;
    let mut htail: u64 = 0;

    let mut sts = [0.0f64; NW];
    let mut oslot = [0usize; NM];
    let mut hslot = [0usize; NH];
    let mut hbase = [0u64; NH];
    let mut sp_slot = [0usize; NSP];
    let mut wd_slot = 0usize;
    let mut wd2_slot = 0usize;
    let mut prev_word_hash: u64 = 0;

    let split = (n as f64 * 0.8) as usize;
    let mut tot = 0.0f64;
    let mut tail = 0.0f64;
    let mut tailn = 0usize;
    let t0 = Instant::now();

    for i in 0..n {
        let prefix = cur;
        // byte-aware orders 0..6 (flat, high-bit hash + tag)
        for k in 0..NM {
            let b = ORDERS[k];
            let l = if byte_pos >= b { b } else { byte_pos };
            let run_ = ((htail & maskb[l]) << phase) | cur;
            let key = (((1u64 << (8 * l + phase)) | run_) << 3) | (phase as u64);
            let h = key.wrapping_mul(MULT);
            let ti = (h >> (64 - obits)) as usize;
            let want = ((h >> (64 - obits - 8)) & 0xFF) as u8;
            if otag[k][ti] != want { otag[k][ti] = want; ocount[k][2 * ti] = 0; ocount[k][2 * ti + 1] = 0; }
            oslot[k] = ti;
            let (n0, n1) = (ocount[k][2 * ti] as f64, ocount[k][2 * ti + 1] as f64);
            sts[k] = stretch((n1 + delta) / (n0 + n1 + d2));
        }
        // high orders (merged hashed)
        for hk in 0..NH {
            let hv = hbase[hk];
            let slot = ((hv.wrapping_mul(2654435761)
                ^ (phase as u64).wrapping_mul(0x9E37_79B1)
                ^ prefix.wrapping_mul(2246822519)) & hmask) as usize * 2;
            hslot[hk] = slot;
            let (n0, n1) = (htab[hk][slot] as f64, htab[hk][slot + 1] as f64);
            sts[NM + hk] = stretch((n1 + delta) / (n0 + n1 + d2));
        }
        // sparse models: non-adjacent byte pairs (capture gaps the contiguous orders miss)
        for j in 0..NSP {
            let (oa, ob) = SPOFF[j];
            let ba = if byte_pos >= oa { hist[byte_pos - oa] as u64 } else { 0 };
            let bb = if byte_pos >= ob { hist[byte_pos - ob] as u64 } else { 0 };
            let sk = ((((((1u64 << phase) | cur) << 8) | ba) << 8 | bb) << 5) | ((j as u64) << 3) | (phase as u64);
            let h = sk.wrapping_mul(MULT);
            let ti = (h >> (64 - obits)) as usize;
            let want = ((h >> (64 - obits - 8)) & 0xFF) as u8;
            if sptag[j][ti] != want { sptag[j][ti] = want; spc[j][2 * ti] = 0; spc[j][2 * ti + 1] = 0; }
            sp_slot[j] = ti;
            let (n0, n1) = (spc[j][2 * ti] as f64, spc[j][2 * ti + 1] as f64);
            sts[NM + NH + j] = stretch((n1 + delta) / (n0 + n1 + d2));
        }
        // word
        let wk = (word_hash << 12) | ((((1u64 << phase) | cur) << 3) | (phase as u64));
        {
            let h = wk.wrapping_mul(MULT);
            let ti = (h >> (64 - obits)) as usize;
            let want = ((h >> (64 - obits - 8)) & 0xFF) as u8;
            if wdtag[ti] != want { wdtag[ti] = want; wdc[2 * ti] = 0; wdc[2 * ti + 1] = 0; }
            wd_slot = ti;
            let (n0, n1) = (wdc[2 * ti] as f64, wdc[2 * ti + 1] as f64);
            sts[NM + NH + NSP] = stretch((n1 + delta) / (n0 + n1 + d2));
        }
        // previous-word model: predict the current word's bits from the WORD BEFORE it (text bigrams)
        let wctx = prev_word_hash.wrapping_mul(0x9E37_79B1).wrapping_add(word_hash.wrapping_mul(2654435761));
        let wk2 = (wctx << 12) | ((((1u64 << phase) | cur) << 3) | (phase as u64));
        {
            let h = wk2.wrapping_mul(MULT);
            let ti = (h >> (64 - obits)) as usize;
            let want = ((h >> (64 - obits - 8)) & 0xFF) as u8;
            if wdtag2[ti] != want { wdtag2[ti] = want; wdc2[2 * ti] = 0; wdc2[2 * ti + 1] = 0; }
            wd2_slot = ti;
            let (n0, n1) = (wdc2[2 * ti] as f64, wdc2[2 * ti + 1] as f64);
            sts[NM + NH + NSP + 1] = stretch((n1 + delta) / (n0 + n1 + d2));
        }
        sts[NM + NH + NSP + 2] = mm.predicted(&hist, phase, byte_pos);
        sts[NM + NH + NSP + 3] = mm2.predicted(&hist, phase, byte_pos);
        sts[NIN] = 1.0;

        let sel = ((phase << 8) | prev_byte as usize) & (NSEL - 1);
        let sel2 = ((phase << 11) | ((prev_byte.wrapping_mul(769) ^ prev2.wrapping_mul(2246822519)) as usize & 2047)) & (NSEL2 - 1);
        let mut d = 0.0;
        for k in 0..NW { d += mixers[sel][k] * sts[k]; }
        let p_sel = squash(d);
        let mut d2 = 0.0;
        for k in 0..NW { d2 += mixers2[sel2][k] * sts[k]; }
        let p_sel2 = squash(d2);
        let mut dg = 0.0;
        for k in 0..NW { dg += gmix[k] * sts[k]; }
        let p_g = squash(dg);
        let ssel = stretch(p_sel);
        let ssel2 = stretch(p_sel2);
        let sg = stretch(p_g);
        let p_mix = squash(final_w[0] * ssel + final_w[1] * sg + final_w[2] * ssel2 + final_w[3]);
        let pa0 = apm1.refine(p_mix, ((prev_byte << 3) | phase as u64) as usize);
        let pa = 0.3 * p_mix + 0.7 * pa0;
        let pb0 = apm2.refine(pa, ((prev_byte.wrapping_mul(769) + prev2.wrapping_mul(31) + phase as u64) & 1023) as usize);
        let mut p = 0.3 * pa + 0.7 * pb0;
        let pc0 = apm3.refine(p, ((cur << 3) | phase as u64) as usize);
        p = 0.3 * p + 0.7 * pc0;
        let pd0 = apm4.refine(p, ((mm.len.min(31) << 3) | phase) as usize);
        p = 0.3 * p + 0.7 * pd0;
        let pe0 = apm5.refine(p, (((word_hash & 511) << 3) | phase as u64) as usize);
        p = 0.3 * p + 0.7 * pe0;
        let pf0 = apm6.refine(p, ((((htail ^ (htail >> 13)) & 511) << 3) | phase as u64) as usize);
        p = 0.3 * p + 0.7 * pf0;
        p = p.clamp(1e-6, 1.0 - 1e-6);

        let y = bits[i];
        let yf = y as f64;
        let cost = -(if y == 1 { p } else { 1.0 - p }).log2();
        tot += cost;
        if i >= split { tail += cost; tailn += 1; }

        // --- updates ---
        let em = yf - p_mix;
        let (gf0, gf1, gf2, gf3) = (em * ssel, em * sg, em * ssel2, em);
        let ord_ = 1.0 - RMS_DECAY;
        final_g[0] = RMS_DECAY * final_g[0] + ord_ * gf0 * gf0;
        final_g[1] = RMS_DECAY * final_g[1] + ord_ * gf1 * gf1;
        final_g[2] = RMS_DECAY * final_g[2] + ord_ * gf2 * gf2;
        final_g[3] = RMS_DECAY * final_g[3] + ord_ * gf3 * gf3;
        final_w[0] += alr_lrf * gf0 / (final_g[0].sqrt() + RMS_EPS);
        final_w[1] += alr_lrf * gf1 / (final_g[1].sqrt() + RMS_EPS);
        final_w[2] += alr_lrf * gf2 / (final_g[2].sqrt() + RMS_EPS);
        final_w[3] += alr_lrf * gf3 / (final_g[3].sqrt() + RMS_EPS);
        let _ = lr_final;
        let e_sel = yf - p_sel;
        let e_sel2 = yf - p_sel2;
        let e_g = yf - p_g;
        {
            let w = &mut mixers[sel];
            let wg = &mut mixers_g[sel];
            for k in 0..NW {
                let gk = e_sel * sts[k];
                wg[k] = RMS_DECAY * wg[k] + ord_ * gk * gk;
                w[k] += alr_lr * gk / (wg[k].sqrt() + RMS_EPS);
            }
        }
        {
            let w = &mut mixers2[sel2];
            let wg = &mut mixers2_g[sel2];
            for k in 0..NW {
                let gk = e_sel2 * sts[k];
                wg[k] = RMS_DECAY * wg[k] + ord_ * gk * gk;
                w[k] += alr_lr * gk / (wg[k].sqrt() + RMS_EPS);
            }
        }
        for k in 0..NW {
            let gk = e_g * sts[k];
            gmix_g[k] = RMS_DECAY * gmix_g[k] + ord_ * gk * gk;
            gmix[k] += alr_lr * gk / (gmix_g[k].sqrt() + RMS_EPS);
        }
        // count bumps + recency halving
        let yi = y as usize;
        for k in 0..NM {
            let s = 2 * oslot[k];
            ocount[k][s + yi] += 1;
            if ocount[k][s + yi] >= CLIMIT { ocount[k][s] = (ocount[k][s] + 1) >> 1; ocount[k][s + 1] = (ocount[k][s + 1] + 1) >> 1; }
        }
        for j in 0..NSP {
            let s = 2 * sp_slot[j]; spc[j][s + yi] += 1;
            if spc[j][s + yi] >= CLIMIT { spc[j][s] = (spc[j][s] + 1) >> 1; spc[j][s + 1] = (spc[j][s + 1] + 1) >> 1; }
        }
        {
            let s = 2 * wd_slot; wdc[s + yi] += 1;
            if wdc[s + yi] >= CLIMIT { wdc[s] = (wdc[s] + 1) >> 1; wdc[s + 1] = (wdc[s + 1] + 1) >> 1; }
        }
        {
            let s = 2 * wd2_slot; wdc2[s + yi] += 1;
            if wdc2[s + yi] >= CLIMIT { wdc2[s] = (wdc2[s] + 1) >> 1; wdc2[s + 1] = (wdc2[s + 1] + 1) >> 1; }
        }
        for hk in 0..NH {
            let s = hslot[hk]; htab[hk][s + yi] += 1;
            if htab[hk][s + yi] >= CLIMIT { htab[hk][s] = (htab[hk][s] + 1) >> 1; htab[hk][s + 1] = (htab[hk][s + 1] + 1) >> 1; }
        }
        apm1.update(yf);
        apm2.update(yf);
        apm3.update(yf);
        apm4.update(yf);
        apm5.update(yf);
        apm6.update(yf);

        cur = (cur << 1) | (y as u64);
        phase += 1;
        if phase == 8 {
            let b = (cur & 0xFF) as u8;
            hist.push(b);
            mm.update_after_byte(&hist, byte_pos);
            mm2.update_after_byte(&hist, byte_pos);
            htail = ((htail << 8) | (b as u64)) & maskb[MAXB];
            if (65..=90).contains(&b) || (97..=122).contains(&b) {
                word_hash = (word_hash.wrapping_mul(131) + ((b | 0x20) as u64)) & 0xFFF_FFFF;
            } else {
                if word_hash != 0 { prev_word_hash = word_hash; } // remember the word that just ended
                word_hash = 0;
            }
            prev2 = prev_byte; prev_byte = b as u64; cur = 0; phase = 0; byte_pos += 1;
            for hk in 0..NH {
                let bb = HORDERS[hk];
                let lo = if byte_pos >= bb { byte_pos - bb } else { 0 };
                let mut hv: u64 = 1469598103934665603;
                for bp in lo..byte_pos { hv = (hv ^ hist[bp] as u64).wrapping_mul(1099511628211); }
                hbase[hk] = hv;
            }
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let whole = tot / n as f64;
    let last = if tailn > 0 { tail / tailn as f64 } else { 0.0 };
    println!("corpus={}  bytes={}  bits={}  obits={}", path, raw.len(), n, obits);
    println!("  blmrs-strong  whole-stream = {:.6}   last-20% = {:.6}  bits/bit   [{:.1}s, {:.1} Mbits/s]",
             whole, last, secs, (n as f64 / 1e6) / secs);
}

/**
 * feature_extractor.js
 * --------------------
 * Exact browser / Android WebView port of python/feature_extractor.py.
 * Computes the 28-dimensional plant pathology feature vector from any 128x128
 * RGBA pixel buffer plus segmentation heatmap overlay for farmer UI feedback.
 */
(function (root) {
  "use strict";

  const IMG_SIZE = 128;

  const FEATURE_NAMES = [
    "r_mean",
    "g_mean",
    "b_mean",
    "r_std",
    "g_std",
    "b_std",
    "exg_mean",
    "exg_std",
    "exr_mean",
    "gli_mean",
    "s_mean",
    "v_mean",
    "healthy_green_frac",
    "yellow_chlorosis_frac",
    "rust_orange_brown_frac",
    "white_pustule_frac",
    "dark_necrotic_frac",
    "tan_straw_frac",
    "hue_dispersion",
    "edge_lesion_energy",
    "stripe_coherence",
    "high_freq_spot_ratio",
    "radial_margin_disease_bias",
    "patch_heterogeneity",
    "leaf_elongation",
    "perimeter_area_ratio",
    "rg_diff_std",
    "dog_ring_energy",
  ];

  function boxBlur(arr, w, h, radius) {
    const stride = w + 1;
    const integral = new Float64Array((h + 1) * stride);
    for (let y = 0; y < h; y++) {
      let rowSum = 0.0;
      const srcRow = y * w;
      const prevIntRow = y * stride;
      const curIntRow = (y + 1) * stride;
      for (let x = 0; x < w; x++) {
        rowSum += arr[srcRow + x];
        integral[curIntRow + x + 1] = integral[prevIntRow + x + 1] + rowSum;
      }
    }

    const out = new Float64Array(w * h);
    for (let y = 0; y < h; y++) {
      const y0 = y - radius > 0 ? y - radius : 0;
      const y1 = (y + radius < h - 1 ? y + radius : h - 1) + 1;
      const r0 = y0 * stride;
      const r1 = y1 * stride;
      const dy = y1 - y0;
      const outRow = y * w;
      for (let x = 0; x < w; x++) {
        const x0 = x - radius > 0 ? x - radius : 0;
        const x1 = (x + radius < w - 1 ? x + radius : w - 1) + 1;
        const sum =
          integral[r1 + x1] -
          integral[r0 + x1] -
          integral[r1 + x0] +
          integral[r0 + x0];
        out[outRow + x] = sum / (dy * (x1 - x0));
      }
    }
    return out;
  }

  function extractFeaturesFromRGBA(rgbaData, width, height) {
    const w = width || IMG_SIZE;
    const h = height || IMG_SIZE;
    const n = w * h;

    const r = new Float64Array(n);
    const g = new Float64Array(n);
    const b = new Float64Array(n);
    const hDeg = new Float64Array(n);
    const s = new Float64Array(n);
    const v = new Float64Array(n);
    const exg = new Float64Array(n);
    const exr = new Float64Array(n);
    const gli = new Float64Array(n);
    const lum = new Float64Array(n);
    const vegSeed = new Float64Array(n);

    for (let i = 0; i < n; i++) {
      const p = i * 4;
      const rv = rgbaData[p] / 255.0;
      const gv = rgbaData[p + 1] / 255.0;
      const bv = rgbaData[p + 2] / 255.0;

      r[i] = rv;
      g[i] = gv;
      b[i] = bv;

      const maxC = Math.max(rv, gv, bv);
      const minC = Math.min(rv, gv, bv);
      const delta = maxC - minC;

      v[i] = maxC;
      s[i] = maxC > 1e-6 ? delta / maxC : 0.0;

      let hue = 0.0;
      if (delta > 1e-6) {
        if (maxC === rv) {
          let m = ((gv - bv) / delta) % 6.0;
          if (m < 0.0) m += 6.0;
          hue = 60.0 * m;
        } else if (maxC === gv) {
          hue = 60.0 * ((bv - rv) / delta + 2.0);
        } else {
          hue = 60.0 * ((rv - gv) / delta + 4.0);
        }
      }
      hDeg[i] = hue;

      const exgVal = 2.0 * gv - rv - bv;
      exg[i] = exgVal;
      exr[i] = 1.4 * rv - gv;
      const gliVal = exgVal / (2.0 * gv + rv + bv + 1e-4);
      gli[i] = gliVal;
      lum[i] = 0.299 * rv + 0.587 * gv + 0.114 * bv;

      const isVeg =
        gliVal > 0.015 ||
        (hue >= 30.0 && hue <= 165.0 && s[i] > 0.13 && v[i] > 0.11);
      vegSeed[i] = isVeg ? 1.0 : 0.0;
    }

    const vegDensity = boxBlur(vegSeed, w, h, 5);
    const leafMask = new Uint8Array(n);
    const leafMaskFloat = new Float64Array(n);
    let nLeaf = 0;

    for (let i = 0; i < n; i++) {
      const isSkyBlue = hDeg[i] >= 175.0 && hDeg[i] <= 265.0 && s[i] > 0.15;
      const isOuterGlare =
        v[i] > 0.96 && s[i] < 0.06 && vegDensity[i] < 0.3;
      const m =
        (vegSeed[i] > 0.5 || vegDensity[i] >= 0.2) &&
        !isSkyBlue &&
        v[i] >= 0.05 &&
        !isOuterGlare;
      if (m) {
        leafMask[i] = 1;
        leafMaskFloat[i] = 1.0;
        nLeaf++;
      }
    }

    let usedFallback = false;
    if (nLeaf < 650) {
      usedFallback = true;
      nLeaf = 0;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const i = y * w + x;
          const ell =
            ((x - 63.5) / 48.0) ** 2 + ((y - 63.5) / 48.0) ** 2 <= 1.0;
          const isSkyBlue = hDeg[i] >= 175.0 && hDeg[i] <= 265.0 && s[i] > 0.15;
          const notGlare = !(v[i] > 0.95 && s[i] < 0.07);
          if (ell && !isSkyBlue && v[i] >= 0.06 && notGlare) {
            leafMask[i] = 1;
            leafMaskFloat[i] = 1.0;
            nLeaf++;
          } else {
            leafMask[i] = 0;
            leafMaskFloat[i] = 0.0;
          }
        }
      }
    }

    if (nLeaf < 60) {
      return {
        leafDetected: false,
        leafPixelRatio: nLeaf / n,
        usedFallbackRoi: usedFallback,
        estimatedSeverityPct: 0.0,
        severityGrade: 0,
        features: new Array(FEATURE_NAMES.length).fill(0.0),
        leafMask: leafMask,
        overlayRGBA: new Uint8ClampedArray(rgbaData),
      };
    }

    const isHealthyPx = new Uint8Array(n);
    const overlayRGBA = new Uint8ClampedArray(n * 4);

    let rSum = 0.0,
      gSum = 0.0,
      bSum = 0.0;
    let exgSum = 0.0,
      exrSum = 0.0,
      gliSum = 0.0,
      sSum = 0.0,
      vSum = 0.0;
    let healthyCnt = 0,
      yellowCnt = 0,
      rustCnt = 0,
      whiteCnt = 0,
      darkCnt = 0,
      tanCnt = 0;
    let satCnt = 0,
      cosSum = 0.0,
      sinSum = 0.0;

    for (let i = 0; i < n; i++) {
      const hue = hDeg[i];
      const sat = s[i];
      const val = v[i];
      const exgVal = exg[i];
      const lumRGB = (r[i] + g[i] + b[i]) / 3.0;

      const hPx =
        hue >= 68.0 && hue <= 165.0 && sat >= 0.16 && val >= 0.16 && exgVal > 0.03;
      const yPx = hue >= 38.0 && hue < 68.0 && sat >= 0.2 && val >= 0.25;
      const ruPx =
        ((hue >= 8.0 && hue < 38.0) || hue >= 345.0) &&
        sat >= 0.2 &&
        val >= 0.16 &&
        val <= 0.88;
      const wPx = val >= 0.72 && sat <= 0.25 && lumRGB >= 0.65;
      const dPx = val <= 0.28 && exgVal < 0.05;
      const tPx =
        hue >= 25.0 &&
        hue < 54.0 &&
        sat >= 0.11 &&
        sat <= 0.38 &&
        val >= 0.4 &&
        val <= 0.85 &&
        !hPx;

      if (hPx) isHealthyPx[i] = 1;

      const p = i * 4;
      if (leafMask[i]) {
        rSum += r[i];
        gSum += g[i];
        bSum += b[i];
        exgSum += exgVal;
        exrSum += exr[i];
        gliSum += gli[i];
        sSum += sat;
        vSum += val;

        if (hPx) healthyCnt++;
        if (yPx) yellowCnt++;
        if (ruPx) rustCnt++;
        if (wPx) whiteCnt++;
        if (dPx) darkCnt++;
        if (tPx) tanCnt++;

        if (sat >= 0.15) {
          const rad = (hue * Math.PI) / 180.0;
          cosSum += Math.cos(rad);
          sinSum += Math.sin(rad);
          satCnt++;
        }

        // Visual Overlay Heatmap for Farmer Review
        if (hPx) {
          overlayRGBA[p] = Math.round(rgbaData[p] * 0.65 + 22 * 0.35);
          overlayRGBA[p + 1] = Math.round(rgbaData[p + 1] * 0.65 + 163 * 0.35);
          overlayRGBA[p + 2] = Math.round(rgbaData[p + 2] * 0.65 + 74 * 0.35);
        } else if (wPx) {
          overlayRGBA[p] = 224;
          overlayRGBA[p + 1] = 242;
          overlayRGBA[p + 2] = 254;
        } else if (yPx || tPx) {
          overlayRGBA[p] = 245;
          overlayRGBA[p + 1] = 158;
          overlayRGBA[p + 2] = 11;
        } else {
          overlayRGBA[p] = 220;
          overlayRGBA[p + 1] = 38;
          overlayRGBA[p + 2] = 38;
        }
        overlayRGBA[p + 3] = 255;
      } else {
        overlayRGBA[p] = Math.round(rgbaData[p] * 0.35);
        overlayRGBA[p + 1] = Math.round(rgbaData[p + 1] * 0.35);
        overlayRGBA[p + 2] = Math.round(rgbaData[p + 2] * 0.35);
        overlayRGBA[p + 3] = 255;
      }
    }

    const rMean = rSum / nLeaf;
    const gMean = gSum / nLeaf;
    const bMean = bSum / nLeaf;
    const exgMean = exgSum / nLeaf;
    const exrMean = exrSum / nLeaf;
    const gliMean = gliSum / nLeaf;
    const sMean = sSum / nLeaf;
    const vMean = vSum / nLeaf;

    let rgSum = 0.0;
    for (let i = 0; i < n; i++) {
      if (leafMask[i]) rgSum += r[i] - g[i];
    }
    const rgMean = rgSum / nLeaf;

    let rVar = 0.0,
      gVar = 0.0,
      bVar = 0.0,
      exgVar = 0.0,
      rgVar = 0.0;
    for (let i = 0; i < n; i++) {
      if (leafMask[i]) {
        rVar += (r[i] - rMean) ** 2;
        gVar += (g[i] - gMean) ** 2;
        bVar += (b[i] - bMean) ** 2;
        exgVar += (exg[i] - exgMean) ** 2;
        rgVar += (r[i] - g[i] - rgMean) ** 2;
      }
    }

    const rStd = Math.sqrt(rVar / nLeaf);
    const gStd = Math.sqrt(gVar / nLeaf);
    const bStd = Math.sqrt(bVar / nLeaf);
    const exgStd = Math.sqrt(exgVar / nLeaf);
    const rgDiffStd = Math.sqrt(rgVar / nLeaf);

    const healthyGreenFrac = healthyCnt / nLeaf;
    const yellowChlorosisFrac = yellowCnt / nLeaf;
    const rustOrangeBrownFrac = rustCnt / nLeaf;
    const whitePustuleFrac = whiteCnt / nLeaf;
    const darkNecroticFrac = darkCnt / nLeaf;
    const tanStrawFrac = tanCnt / nLeaf;

    let hueDispersion = 0.0;
    if (satCnt > 10) {
      const cMean = cosSum / satCnt;
      const sSin = sinSum / satCnt;
      hueDispersion = 1.0 - Math.sqrt(cMean * cMean + sSin * sSin);
    }

    const gx = new Float64Array(n);
    const gy = new Float64Array(n);
    const gx2 = new Float64Array(n);
    const gy2 = new Float64Array(n);
    const gxy = new Float64Array(n);
    let edgeSum = 0.0;

    for (let y = 0; y < h; y++) {
      const ym = y > 0 ? y - 1 : 0;
      const yp = y < h - 1 ? y + 1 : h - 1;
      for (let x = 0; x < w; x++) {
        const xm = x > 0 ? x - 1 : 0;
        const xp = x < w - 1 ? x + 1 : w - 1;
        const i = y * w + x;
        const dx = 0.5 * (lum[y * w + xp] - lum[y * w + xm]);
        const dy = 0.5 * (lum[yp * w + x] - lum[ym * w + x]);
        gx[i] = dx;
        gy[i] = dy;
        gx2[i] = dx * dx;
        gy2[i] = dy * dy;
        gxy[i] = dx * dy;
        if (leafMask[i]) {
          edgeSum += 0.5 * (Math.abs(dx) + Math.abs(dy));
        }
      }
    }
    const edgeLesionEnergy = edgeSum / nLeaf;

    const jxx = boxBlur(gx2, w, h, 3);
    const jyy = boxBlur(gy2, w, h, 3);
    const jxy = boxBlur(gxy, w, h, 3);
    let cohWeightedSum = 0.0;
    let weightSum = 0.0;
    for (let i = 0; i < n; i++) {
      if (leafMask[i]) {
        const tr = jxx[i] + jyy[i];
        if (tr > 1e-4) {
          const coh =
            Math.sqrt((jxx[i] - jyy[i]) ** 2 + 4.0 * jxy[i] ** 2) /
            (tr + 1e-6);
          cohWeightedSum += coh * tr;
          weightSum += tr;
        }
      }
    }
    const stripeCoherence =
      weightSum > 0.0 ? cohWeightedSum / (weightSum + 1e-9) : 0.0;

    const exgBlur5 = boxBlur(exg, w, h, 2);
    let spotCnt = 0;
    for (let i = 0; i < n; i++) {
      if (leafMask[i] && Math.abs(exg[i] - exgBlur5[i]) > 0.07) {
        spotCnt++;
      }
    }
    const highFreqSpotRatio = spotCnt / nLeaf;

    const maskDensityR3 = boxBlur(leafMaskFloat, w, h, 3);
    let nMargin = 0,
      nMarginLesion = 0,
      nCore = 0,
      nCoreLesion = 0;
    for (let i = 0; i < n; i++) {
      if (leafMask[i]) {
        const isLesion = isHealthyPx[i] === 0;
        if (maskDensityR3[i] < 0.88) {
          nMargin++;
          if (isLesion) nMarginLesion++;
        } else {
          nCore++;
          if (isLesion) nCoreLesion++;
        }
      }
    }
    const marginFrac = nMargin > 0 ? nMarginLesion / nMargin : 0.0;
    const coreFrac = nCore > 0 ? nCoreLesion / nCore : 0.0;
    const radialMarginDiseaseBias = marginFrac - coreFrac;

    const blockMeans = [];
    for (let by = 0; by < 4; by++) {
      for (let bx = 0; bx < 4; bx++) {
        let bSum = 0.0,
          bCnt = 0;
        for (let y = by * 32; y < (by + 1) * 32; y++) {
          for (let x = bx * 32; x < (bx + 1) * 32; x++) {
            const i = y * w + x;
            if (leafMask[i]) {
              bSum += exg[i];
              bCnt++;
            }
          }
        }
        if (bCnt >= 16) blockMeans.push(bSum / bCnt);
      }
    }
    let patchHeterogeneity = 0.0;
    if (blockMeans.length >= 2) {
      const bmMean =
        blockMeans.reduce((a, c) => a + c, 0.0) / blockMeans.length;
      const bmVar =
        blockMeans.reduce((a, c) => a + (c - bmMean) ** 2, 0.0) /
        blockMeans.length;
      patchHeterogeneity = Math.sqrt(bmVar);
    }

    // PCA Elongation & 4-connected Perimeter
    let xCoordSum = 0.0,
      yCoordSum = 0.0,
      perimeterPx = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (leafMask[i]) {
          xCoordSum += x;
          yCoordSum += y;
          const interior =
            y > 0 &&
            leafMask[i - w] &&
            y < h - 1 &&
            leafMask[i + w] &&
            x > 0 &&
            leafMask[i - 1] &&
            x < w - 1 &&
            leafMask[i + 1];
          if (!interior) perimeterPx++;
        }
      }
    }
    const xBar = xCoordSum / nLeaf;
    const yBar = yCoordSum / nLeaf;
    let cxx = 0.0,
      cyy = 0.0,
      cxy = 0.0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (leafMask[i]) {
          const dx = x - xBar;
          const dy = y - yBar;
          cxx += dx * dx;
          cyy += dy * dy;
          cxy += dx * dy;
        }
      }
    }
    cxx /= nLeaf;
    cyy /= nLeaf;
    cxy /= nLeaf;
    const trCov = cxx + cyy;
    const discCov = Math.sqrt((cxx - cyy) ** 2 + 4.0 * cxy * cxy);
    const lam1 = 0.5 * (trCov + discCov);
    const lam2 = Math.max(0.0, 0.5 * (trCov - discCov));
    const leafElongation = 1.0 - lam2 / (lam1 + 1e-6);
    const perimeterAreaRatio = perimeterPx / (Math.sqrt(nLeaf) + 1e-6);

    const lumB1 = boxBlur(lum, w, h, 1);
    const lumB4 = boxBlur(lum, w, h, 4);
    let dogLesionSum = 0.0,
      nLesion = 0,
      dogLeafSum = 0.0;
    for (let i = 0; i < n; i++) {
      if (leafMask[i]) {
        const dVal = Math.abs(lumB1[i] - lumB4[i]);
        dogLeafSum += dVal;
        if (isHealthyPx[i] === 0) {
          dogLesionSum += dVal;
          nLesion++;
        }
      }
    }
    const dogRingEnergy =
      nLesion > 12 ? dogLesionSum / nLesion : dogLeafSum / nLeaf;

    const features = [
      rMean,
      gMean,
      bMean,
      rStd,
      gStd,
      bStd,
      exgMean,
      exgStd,
      exrMean,
      gliMean,
      sMean,
      vMean,
      healthyGreenFrac,
      yellowChlorosisFrac,
      rustOrangeBrownFrac,
      whitePustuleFrac,
      darkNecroticFrac,
      tanStrawFrac,
      hueDispersion,
      edgeLesionEnergy,
      stripeCoherence,
      highFreqSpotRatio,
      radialMarginDiseaseBias,
      patchHeterogeneity,
      leafElongation,
      perimeterAreaRatio,
      rgDiffStd,
      dogRingEnergy,
    ];

    const rawLesionPct = 100.0 * (1.0 - healthyGreenFrac);
    const estSeverity =
      Math.round(Math.max(0.0, Math.min(100.0, rawLesionPct)) * 10) / 10;
    let grade = 0;
    if (estSeverity > 75.0) grade = 5;
    else if (estSeverity > 50.0) grade = 4;
    else if (estSeverity > 25.0) grade = 3;
    else if (estSeverity > 10.0) grade = 2;
    else if (estSeverity > 2.0) grade = 1;

    return {
      leafDetected: true,
      leafPixelRatio: nLeaf / n,
      usedFallbackRoi: usedFallback,
      estimatedSeverityPct: estSeverity,
      severityGrade: grade,
      features: features,
      leafMask: leafMask,
      overlayRGBA: overlayRGBA,
    };
  }

  const api = {
    IMG_SIZE,
    FEATURE_NAMES,
    extractFeaturesFromRGBA,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.RabiFeatureExtractor = api;
  }
})(typeof window !== "undefined" ? window : globalThis);

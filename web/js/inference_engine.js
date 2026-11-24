/**
 * inference_engine.js
 * -------------------
 * Zero-dependency client-side Machine Learning & Artificial Neural Network engine.
 * Runs inside Android WebView and desktop browsers offline using weights
 * exported by python/train_and_evaluate.py.
 *
 * Models executed natively:
 *   1. MLP Artificial Neural Network (28 -> 64 -> 32 -> 12, ReLU + Softmax)
 *   2. Deep Regularized ANN (28 -> 128 -> 64 -> 32 -> 12, ReLU + Softmax)
 *   3. Random Forest (45 Decision Trees ensemble)
 *   4. RBF Kernel Support Vector Machine (12-class OvO + Platt probability coupling)
 *   5. Hybrid Weighted Ensemble combining all four models
 */
(function (root) {
  "use strict";

  function standardizeFeatures(features, scaler) {
    const out = new Float64Array(features.length);
    for (let i = 0; i < features.length; i++) {
      const sc = scaler.scale[i] > 1e-12 ? scaler.scale[i] : 1.0;
      out[i] = (features[i] - scaler.mean[i]) / sc;
    }
    return out;
  }

  function softmax(logits) {
    let maxVal = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxVal) maxVal = logits[i];
    }
    const exps = new Array(logits.length);
    let sum = 0.0;
    for (let i = 0; i < logits.length; i++) {
      const e = Math.exp(logits[i] - maxVal);
      exps[i] = e;
      sum += e;
    }
    for (let i = 0; i < logits.length; i++) {
      exps[i] /= sum;
    }
    return exps;
  }

  function predictMLP(xScaled, mlpModel) {
    let act = xScaled;
    const nLayers = mlpModel.layers.length;
    for (let l = 0; l < nLayers; l++) {
      const layer = mlpModel.layers[l];
      const W = layer.W;
      const b = layer.b;
      const inDim = W.length;
      const outDim = b.length;
      const next = new Float64Array(outDim);

      for (let j = 0; j < outDim; j++) {
        next[j] = b[j];
      }
      for (let i = 0; i < inDim; i++) {
        const ai = act[i];
        const wi = W[i];
        for (let j = 0; j < outDim; j++) {
          next[j] += ai * wi[j];
        }
      }
      if (l < nLayers - 1) {
        for (let j = 0; j < outDim; j++) {
          if (next[j] < 0.0) next[j] = 0.0; // ReLU
        }
      }
      act = next;
    }
    return softmax(act);
  }

  function predictRandomForest(xScaled, rfModel, nClasses) {
    const probs = new Array(nClasses).fill(0.0);
    const trees = rfModel.trees;
    const nTrees = trees.length;
    for (let t = 0; t < nTrees; t++) {
      const tree = trees[t];
      let node = 0;
      while (tree.children_left[node] !== -1) {
        const fIdx = tree.feature[node];
        node =
          xScaled[fIdx] <= tree.threshold[node]
            ? tree.children_left[node]
            : tree.children_right[node];
      }
      const leafProbs = tree.probs[node];
      for (let c = 0; c < nClasses; c++) {
        probs[c] += leafProbs[c];
      }
    }
    for (let c = 0; c < nClasses; c++) {
      probs[c] /= nTrees;
    }
    return probs;
  }

  function multiclassProbabilityCoupling(k, r) {
    // Wu, Lin, and Weng (2004) Algorithm 2 matching LIBSVM / scikit-learn
    const p = new Array(k).fill(1.0 / k);
    const Q = [];
    for (let t = 0; t < k; t++) {
      Q.push(new Array(k).fill(0.0));
      Q[t][t] = 0.0;
      for (let j = 0; j < t; j++) {
        Q[t][t] += r[j][t] * r[j][t];
        Q[t][j] = Q[j][t];
      }
      for (let j = t + 1; j < k; j++) {
        Q[t][t] += r[j][t] * r[j][t];
        Q[t][j] = -r[j][t] * r[t][j];
      }
    }

    const Qp = new Array(k).fill(0.0);
    const maxIter = Math.max(100, k);
    const eps = 0.005 / k;

    for (let iter = 0; iter < maxIter; iter++) {
      let pQp = 0.0;
      for (let t = 0; t < k; t++) {
        Qp[t] = 0.0;
        for (let j = 0; j < k; j++) {
          Qp[t] += Q[t][j] * p[j];
        }
        pQp += p[t] * Qp[t];
      }
      let maxError = 0.0;
      for (let t = 0; t < k; t++) {
        const err = Math.abs(Qp[t] - pQp);
        if (err > maxError) maxError = err;
      }
      if (maxError < eps) break;

      for (let t = 0; t < k; t++) {
        const diff = (-Qp[t] + pQp) / Q[t][t];
        p[t] += diff;
        pQp = (pQp + diff * (diff * Q[t][t] + 2.0 * Qp[t])) / (1.0 + diff) / (1.0 + diff);
        for (let j = 0; j < k; j++) {
          Qp[j] = (Qp[j] + diff * Q[t][j]) / (1.0 + diff);
          p[j] /= 1.0 + diff;
        }
      }
    }
    return p;
  }

  function predictRbfSvm(xScaled, svmModel, nClasses) {
    const sv = svmModel.support_vectors;
    const nSv = sv.length;
    const gamma = svmModel.gamma;
    const kVal = new Float64Array(nSv);

    for (let s = 0; s < nSv; s++) {
      const svRow = sv[s];
      let sqDist = 0.0;
      for (let d = 0; d < xScaled.length; d++) {
        const diff = xScaled[d] - svRow[d];
        sqDist += diff * diff;
      }
      kVal[s] = Math.exp(-gamma * sqDist);
    }

    const starts = new Array(nClasses);
    let offset = 0;
    for (let c = 0; c < nClasses; c++) {
      starts[c] = offset;
      offset += svmModel.n_support[c];
    }

    const pairwise = [];
    for (let i = 0; i < nClasses; i++) {
      pairwise.push(new Array(nClasses).fill(0.0));
    }

    const dualCoef = svmModel.dual_coef;
    const intercept = svmModel.intercept;
    const probA = svmModel.probA;
    const probB = svmModel.probB;
    const minProb = 1e-7;

    let pIdx = 0;
    for (let i = 0; i < nClasses; i++) {
      for (let j = i + 1; j < nClasses; j++) {
        let sum = intercept[pIdx];
        const si = starts[i];
        const ci = svmModel.n_support[i];
        const sj = starts[j];
        const cj = svmModel.n_support[j];

        const coefI = dualCoef[j - 1];
        const coefJ = dualCoef[i];

        for (let k = 0; k < ci; k++) {
          sum += coefI[si + k] * kVal[si + k];
        }
        for (let k = 0; k < cj; k++) {
          sum += coefJ[sj + k] * kVal[sj + k];
        }

        const fApB = sum * probA[pIdx] + probB[pIdx];
        let pIJ =
          fApB >= 0.0
            ? Math.exp(-fApB) / (1.0 + Math.exp(-fApB))
            : 1.0 / (1.0 + Math.exp(fApB));
        pIJ = Math.min(Math.max(pIJ, minProb), 1.0 - minProb);
        pairwise[i][j] = pIJ;
        pairwise[j][i] = 1.0 - pIJ;
        pIdx++;
      }
    }

    return multiclassProbabilityCoupling(nClasses, pairwise);
  }

  function applyCropFilter(probs, classes, cropFilter) {
    if (!cropFilter || cropFilter === "all") return probs.slice();
    const prefix = cropFilter.toLowerCase() + "_";
    const masked = new Array(probs.length);
    let sum = 0.0;
    for (let i = 0; i < classes.length; i++) {
      if (classes[i].startsWith(prefix)) {
        masked[i] = probs[i];
        sum += probs[i];
      } else {
        masked[i] = 0.0;
      }
    }
    if (sum > 1e-12) {
      for (let i = 0; i < masked.length; i++) masked[i] /= sum;
      return masked;
    }
    return probs.slice();
  }

  function topPrediction(probs, classes) {
    let bestIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[bestIdx]) bestIdx = i;
    }
    return {
      classIndex: bestIdx,
      className: classes[bestIdx],
      confidence: probs[bestIdx],
    };
  }

  function predictAllModels(features, modelBundle, cropFilter) {
    const classes = modelBundle.classes;
    const nClasses = classes.length;
    const xScaled = standardizeFeatures(features, modelBundle.scaler);

    const probsAnnMlp = predictMLP(xScaled, modelBundle.ann_mlp);
    const probsAnnDeep = predictMLP(xScaled, modelBundle.ann_deep);
    const probsRf = predictRandomForest(xScaled, modelBundle.random_forest, nClasses);
    const probsSvm = predictRbfSvm(xScaled, modelBundle.svm_rbf, nClasses);

    const w = modelBundle.ensemble_weights;
    const probsEnsemble = new Array(nClasses).fill(0.0);
    for (let c = 0; c < nClasses; c++) {
      probsEnsemble[c] =
        w.ann_deep * probsAnnDeep[c] +
        w.ann_mlp * probsAnnMlp[c] +
        w.svm_rbf * probsSvm[c] +
        w.random_forest * probsRf[c];
    }

    const rawModels = {
      ann_deep: probsAnnDeep,
      ann_mlp: probsAnnMlp,
      random_forest: probsRf,
      svm_rbf: probsSvm,
      hybrid_ensemble: probsEnsemble,
    };

    const filtered = {};
    const winners = {};
    for (const key of Object.keys(rawModels)) {
      const fProbs = applyCropFilter(rawModels[key], classes, cropFilter);
      filtered[key] = fProbs;
      winners[key] = topPrediction(fProbs, classes);
    }

    return {
      xScaled: Array.from(xScaled),
      rawProbs: rawModels,
      filteredProbs: filtered,
      predictions: winners,
    };
  }

  const api = {
    standardizeFeatures,
    predictMLP,
    predictRandomForest,
    predictRbfSvm,
    predictAllModels,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.RabiInferenceEngine = api;
  }
})(typeof window !== "undefined" ? window : globalThis);

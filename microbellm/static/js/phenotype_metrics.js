(function(window) {
    'use strict';

    const MISSING_TOKENS = new Set(['n/a', 'na', 'null', 'none', 'nan', 'undefined', '-', 'unknown', 'missing']);

    function normalizeValue(value) {
        if (value === null || value === undefined || value === '') {
            return null;
        }

        const strValue = String(value).trim().toLowerCase();
        if (!strValue || MISSING_TOKENS.has(strValue)) {
            return null;
        }

        if (strValue.includes(',') || strValue.includes(';')) {
            const parts = strValue.split(/[,;]/).map(s => s.trim()).filter(Boolean);
            return parts.sort().join(',');
        }

        return strValue;
    }

    function toBoolean(value) {
        if (value === null || value === undefined) return null;
        if (typeof value === 'boolean') return value;
        const s = String(value).trim().toLowerCase();
        if (['true', '1', 'yes', 't', 'y'].includes(s)) return true;
        if (['false', '0', 'no', 'f', 'n'].includes(s)) return false;
        return null;
    }

    function computeMetrics(preds, truths) {
        const mapped = preds.map((p, i) => [toBoolean(p), toBoolean(truths[i])])
                            .filter(([p, t]) => p !== null && t !== null);
        const allBinary = mapped.length > 0 && mapped.length === preds.length;

        if (allBinary) {
            let tp = 0, tn = 0, fp = 0, fn = 0;
            for (const [p, t] of mapped) {
                if (t && p) tp++;
                else if (!t && !p) tn++;
                else if (!t && p) fp++;
                else fn++;
            }
            const sens = tp + fn ? tp / (tp + fn) : 0;
            const spec = tn + fp ? tn / (tn + fp) : 0;
            const prec = tp + fp ? tp / (tp + fp) : 0;
            const rec = sens;
            const f1 = (prec + rec) > 0 ? (2 * prec * rec) / (prec + rec) : 0;
            return {
                balancedAcc: (sens + spec) / 2,
                precision: prec,
                recall: rec,
                f1,
                sampleSize: mapped.length,
                confusionMatrix: [[tn, fp], [fn, tp]],
                labels: ['False', 'True']
            };
        }

        const labels = Array.from(new Set([...truths, ...preds].map(v => String(v)))).sort();
        if (labels.length === 0) {
            return {
                balancedAcc: NaN,
                precision: NaN,
                recall: NaN,
                f1: NaN,
                sampleSize: 0,
                confusionMatrix: [],
                labels: []
            };
        }

        const conf = Object.fromEntries(labels.map(r => [r, Object.fromEntries(labels.map(c => [c, 0]))]));
        for (let i = 0; i < truths.length; i++) {
            const truth = String(truths[i]);
            const pred = String(preds[i]);
            if (conf[truth] && conf[truth][pred] !== undefined) {
                conf[truth][pred] += 1;
            }
        }

        let recallSum = 0;
        let precSum = 0;
        let f1Sum = 0;

        labels.forEach(label => {
            const tp = conf[label][label];
            let fn = 0;
            let fp = 0;

            labels.forEach(other => {
                if (other !== label) {
                    fn += conf[label][other];
                    fp += conf[other][label];
                }
            });

            const rec = tp + fn ? tp / (tp + fn) : 0;
            const pre = tp + fp ? tp / (tp + fp) : 0;
            const f1 = (pre + rec) > 0 ? (2 * pre * rec) / (pre + rec) : 0;

            recallSum += rec;
            precSum += pre;
            f1Sum += f1;
        });

        const confusionMatrixArray = labels.map(r => labels.map(c => conf[r][c]));
        return {
            balancedAcc: recallSum / labels.length,
            precision: precSum / labels.length,
            recall: recallSum / labels.length,
            f1: f1Sum / labels.length,
            sampleSize: truths.length,
            confusionMatrix: confusionMatrixArray,
            labels
        };
    }

    function calculateMetrics(predictions, groundTruthMap, options = {}) {
        if (!Array.isArray(predictions) || predictions.length === 0) {
            return [];
        }

        const fields = options.fields || null;
        let phenotypes = Array.isArray(options.phenotypes) ? options.phenotypes.slice() : null;
        if (!phenotypes && fields) {
            phenotypes = Object.keys(fields);
        }
        if (!phenotypes) {
            const sample = predictions[0];
            phenotypes = Object.keys(sample || {}).filter(key => !['model', 'binomial_name'].includes(key));
        }

        const exclude = new Set(options.excludePhenotypes || []);
        phenotypes = phenotypes.filter(p => !exclude.has(p));

        const minSampleSize = typeof options.minSampleSize === 'number' ? options.minSampleSize : 0;

        const predictionsByModel = new Map();
        predictions.forEach(pred => {
            const model = pred.model;
            if (!model) return;
            if (!predictionsByModel.has(model)) {
                predictionsByModel.set(model, []);
            }
            predictionsByModel.get(model).push(pred);
        });

        const results = [];

        phenotypes.forEach(phenotype => {
            predictionsByModel.forEach((rows, model) => {
                const truths = [];
                const predsArray = [];

                rows.forEach(pred => {
                    const species = pred.binomial_name ? pred.binomial_name.toLowerCase() : null;
                    if (!species) return;
                    const gt = groundTruthMap[species];
                    if (!gt) return;
                    const truthVal = normalizeValue(gt[phenotype]);
                    const predVal = normalizeValue(pred[phenotype]);
                    if (truthVal !== null && predVal !== null) {
                        truths.push(truthVal);
                        predsArray.push(predVal);
                    }
                });

                if (truths.length >= minSampleSize) {
                    const metrics = computeMetrics(predsArray, truths);
                    results.push({
                        model,
                        phenotype,
                        balancedAcc: metrics.balancedAcc,
                        precision: metrics.precision,
                        recall: metrics.recall,
                        f1: metrics.f1,
                        sampleSize: metrics.sampleSize,
                        confusionMatrix: metrics.confusionMatrix,
                        labels: metrics.labels
                    });
                }
            });
        });

        return results;
    }

    window.PhenotypeMetrics = {
        normalizeValue,
        computeMetrics,
        calculateMetrics
    };
})(window);

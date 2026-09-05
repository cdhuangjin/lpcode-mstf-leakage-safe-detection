"""Ledger-derived table and figure data; frozen intervals are explicitly retained."""
import csv
import itertools
import json
import math
from statistics import stdev
from audit_manuscript_evidence import SOURCES, LANGUAGES, SEEDS, LLMS, GATE_ROWS, load_ledger, paired_stats, select, sha256
from reproduce import write_json


def validate_mechanism(summary, rows):
    """Compare two frozen aggregate views; this does not recompute importance."""
    environments = {'clean', 'combined_attack', 'cross_language', 'unseen_llm'}
    groups = {block + ':' + family for block in ('human_absolute', 'candidate_absolute', 'delta', 'relative_delta')
              for family in ('original_style', 'lexical', 'structural_syntax', 'formatting_layout')}
    expected = set(itertools.product(environments, groups))
    metrics = {'gain_mean', 'gain_sd', 'permutation_mean', 'permutation_sd'}
    fields = metrics | {'environment', 'frozen_registry_sha256', 'group', 'n_folds'}
    if set(summary['rankings']) != environments:
        raise ValueError('mechanism summary environment coverage')
    index = {}
    for env, ranking in summary['rankings'].items():
        for row in ranking['group_rows']:
            if set(row) != metrics | {'group', 'n_folds'}:
                raise ValueError('mechanism summary schema')
            key = (env, row['group'])
            if key in index: raise ValueError('mechanism summary duplicate group')
            index[key] = row
    if set(index) != expected:
        raise ValueError('mechanism summary group coverage')
    observed = set()
    for row in rows:
        if set(row) != fields: raise ValueError('mechanism aggregate schema')
        key = (row['environment'], row['group'])
        if key in observed: raise ValueError('mechanism aggregate duplicate group')
        observed.add(key)
        if key not in index: raise ValueError('mechanism aggregate unexpected group')
        if row['frozen_registry_sha256'] != summary['frozen_registry_sha256']:
            raise ValueError('mechanism aggregate registry binding')
        for metric in metrics:
            value = float(row[metric])
            if not math.isfinite(value) or value != index[key][metric]:
                raise ValueError('mechanism aggregate mismatch: ' + metric)
        if int(row['n_folds']) != index[key]['n_folds']:
            raise ValueError('mechanism aggregate fold count mismatch')
    if observed != expected: raise ValueError('mechanism aggregate group coverage')


def manuscript_table(name, rows):
    def ci(r): return f"[{r['ci_low_pp']:.3f}, {r['ci_high_pp']:.3f}]"
    def scores(r): return [f"{r['baseline']:.4f}", f"{r['method']:.4f}", f"{r['delta_pp']:+.3f}"]
    if name == 'table2':
        head = ['Gate / comparison', 'Baseline F1', 'Method F1', 'Difference (pp)', '95% CI (pp)']
        body = [[GATE_ROWS[i], *scores(r), ci(r)] for i, r in enumerate(rows)]
    elif name == 'table3':
        head = ['Variant', 'Features', 'Representation', 'Dimensions', 'Clean F1', 'Held-out F1']
        reps = ['Endpoints', 'Endpoints + signed difference', 'Endpoints', 'Signed difference only', 'Endpoints + signed difference', 'Endpoints + signed + relative']
        body = [[r['variant'], 'Original 10' if i < 2 else 'Enhanced 28', reps[i], r['dimensions'], f"{r['clean_f1']:.4f}", f"{r['unseen_f1']:.4f}"] for i, r in enumerate(rows)]
    elif name == 'table4':
        head = ['Held-out generator', 'LPcode original', 'Full MSTF', 'Difference (pp)', '95% interval (pp)']
        labels = dict(zip(LLMS, ['GPT-3.5', 'Gemini-Pro', 'WizardCoder-33B', 'DeepSeek-Coder-33B']))
        body = [[labels[r['generator']], *scores(r), ci(r)] for r in rows]
    elif name == 'table5':
        head = ['Transformation', 'LPcode original', 'Full MSTF', 'Difference (pp)', 'Drops: original / MSTF (pp)']
        body = [[r['condition'].replace('_', ' ').capitalize(), *scores(r), f"{r['lpcode_original_drop_pp']:.3f} / {r['mstf_drop_pp']:.3f}"] for r in rows]
    elif name == 'table6':
        head = ['Held-out language', 'LPcode original: mean ± SD', 'MSTF: mean ± SD', 'Difference (pp)']
        labels = dict(zip(LANGUAGES, ['C', 'C++', 'Java', 'Python']))
        body = [[labels[r['language']], f"{r['baseline']:.4f} ± {r['baseline_sample_sd']:.4f}", f"{r['method']:.4f} ± {r['method_sample_sd']:.4f}", f"{r['delta_pp']:+.3f}"] for r in rows]
    elif name == 'table7':
        head = ['Negatives', 'A0 F1', 'A1 F1', 'Difference (pp)', '95% CI (pp)']
        body = [[r['negative_mode'], *scores(r), ci(r)] for r in rows]
    else:
        raise ValueError('unknown manuscript table')
    return '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in [head, ['---'] * len(head), *body]) + '\n'


def run(root, output, mode):
    manifest = json.loads((root / 'FILE_MANIFEST.json').read_text())['files']
    hashes = {r['path']: r['sha256'] for r in manifest}
    if len(hashes) != len(manifest):
        raise ValueError('duplicate manifest paths')
    provenance = {'interval_policy': 'Frozen intervals retained, not rebootstrap estimates.',
                  'metric_table_policy': 'F1 means, paired differences, clean-to-attack drops and sample SD in metric tables and their figure CSVs are recomputed from row ledgers.',
                  'mechanism_policy': 'Mechanism feature-family importance is retained frozen aggregate data; no importance refits or per-fold importance recomputation claimed. Summary groups are cross-checked against the hash-verified grouped_importance.csv for exact schema, coverage and values.',
                  'sources': {}}

    def read(path):
        data = (root / path).read_bytes()
        if sha256(data) != hashes.get(path):
            raise ValueError('source hash mismatch: ' + path)
        provenance['sources'][path] = hashes[path]
        return json.loads(data) if path.endswith('.json') else data.decode()

    ledgers = {}
    for name, path in SOURCES.items():
        ledgers[name] = load_ledger(root, path, hashes)
        provenance['sources'][path] = hashes[path]
    a_path = 'results/05_mechanism_analysis/ablation_summary.json'
    b_path = 'results/02_unseen_llm/summary.json'
    c_path = 'results/03_style_attack/summary.json'
    d_path = 'repro/2502.17749/v1/results/04_cross_language/summary.json'
    a, b, c, d = [read(p) for p in (a_path, b_path, c_path, d_path)]
    standard_keys = ('language', 'seed', 'fold')
    standard = set(itertools.product(LANGUAGES, SEEDS, range(5)))
    unseen_keys = standard_keys + ('heldout_llm',)
    unseen = set(itertools.product(LANGUAGES, SEEDS, range(5), LLMS))
    language_keys = ('heldout_language', 'seed')
    language_cells = set(itertools.product(LANGUAGES, SEEDS))

    def stats(left, right, keys=standard_keys, cells=standard):
        s = paired_stats(left, right, keys, cells)
        return {k: s[k] for k in ('baseline', 'method', 'delta_pp', 'cells')}

    def interval(row, frozen, expected, path):
        if not math.isclose(row['delta_pp'] / 100, expected, abs_tol=1e-12):
            raise ValueError('frozen summary mean inconsistent: ' + path)
        row.update(ci_low_pp=100 * frozen['low'], ci_high_pp=100 * frozen['high'], ci_source=path)
        return row

    def equal(actual, expected, label):
        if not math.isclose(actual, expected, abs_tol=1e-12):
            raise ValueError('summary inconsistent: ' + label)

    def absolute(row, frozen, label):
        equal(row['baseline'], frozen['lpcode_original']['f1_mean'], label + ' baseline')
        equal(row['method'], frozen['mstf']['f1_mean'], label + ' method')

    def save(name, rows):
        with (output / (name + '.csv')).open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        if name.startswith('table'):
            (output / (name + '.md')).write_text(manuscript_table(name, rows), encoding='utf-8')
        return rows

    main = []
    frozen_main = [a['environments']['clean']['overall']['C1'], b['paired_mstf_minus_lpcode']['overall'], c['paired_mstf_minus_lpcode']['by_condition']['combined'], d['paired_mstf_minus_lpcode']]
    mean_keys = ['mean_delta_f1', 'macro_holdout_language_mean_delta_f1', 'macro_language_mean_delta_f1', 'overall_equal_language_mean_delta_f1']
    for i, gate in enumerate('ABCD'):
        rows = ledgers[gate]
        if gate == 'A':
            left, right = select(rows, model='xgb', representation='concat'), select(rows, model='xgb', representation='concat_delta')
        else:
            if gate == 'C': rows = select(rows, condition='combined')
            left, right = select(rows, method='lpcode_original'), select(rows, method='mstf')
        keys, cells = (unseen_keys, unseen) if gate == 'B' else (language_keys, language_cells) if gate == 'D' else (standard_keys, standard)
        row = dict(gate=gate, **stats(left, right, keys, cells))
        main.append(interval(row, frozen_main[i]['ci_95'], frozen_main[i][mean_keys[i]], [a_path, b_path, c_path, d_path][i]))
    variants = []
    for i, dimension in enumerate([20, 30, 56, 28, 84, 112]):
        row = {'variant': f'A{i}', 'dimensions': dimension}
        for env in ('clean', 'unseen'):
            rows = select(ledgers['ablation'], environment=env)
            keys, cells = (standard_keys, standard) if env == 'clean' else (unseen_keys, unseen)
            row[env + '_f1'] = stats(select(rows, method='A0'), select(rows, method=f'A{i}'), keys, cells)['method']
        variants.append(row)
    if mode in ('table2', 'all-saved'): save('table2', main)
    if mode in ('table3', 'all-saved'): save('table3', variants)
    if mode == 'all-saved':
        contrasts = []
        for env in ('clean', 'unseen'):
            rows = select(ledgers['ablation'], environment=env)
            keys, cells = (standard_keys, standard) if env == 'clean' else (unseen_keys, unseen)
            for name, frozen in a['environments'][env]['overall'].items():
                row = dict(environment=env, contrast=name, **stats(select(rows, method=frozen['right']), select(rows, method=frozen['left']), keys, cells))
                contrasts.append(interval(row, frozen['ci_95'], frozen['mean_delta_f1'], a_path))
        generators = []
        for generator in LLMS:
            rows = select(ledgers['B'], heldout_llm=generator)
            frozen = b['paired_mstf_minus_lpcode']['by_holdout'][generator]
            row = dict(generator=generator, **stats(select(rows, method='lpcode_original'), select(rows, method='mstf')))
            absolute(row, b['macro_language_summaries'][generator], generator)
            generators.append(interval(row, frozen['ci_95'], frozen['macro_language_mean_delta_f1'], b_path))
        attacks = []
        clean = select(ledgers['C'], condition='clean')
        for condition in ('clean', 'comment_removal', 'identifier_rename', 'format_normalization', 'comment_injection', 'combined'):
            rows = select(ledgers['C'], condition=condition)
            frozen = c['paired_mstf_minus_lpcode']['by_condition'][condition]
            row = dict(condition=condition, **stats(select(rows, method='lpcode_original'), select(rows, method='mstf')))
            absolute(row, c['macro_language_summaries'][condition], condition)
            interval(row, frozen['ci_95'], frozen['macro_language_mean_delta_f1'], c_path)
            for method in ('lpcode_original', 'mstf'):
                drop = stats(select(rows, method=method), select(clean, method=method))['delta_pp']
                row[method + '_drop_pp'] = drop
                if condition != 'clean':
                    f = c['clean_to_attack_drops']['by_condition'][condition][method]
                    if not math.isclose(drop / 100, f['macro_language_mean_drop_f1'], abs_tol=1e-12): raise ValueError('drop summary mismatch')
                    row[method + '_drop_ci_low_pp'] = 100 * f['ci_95']['low']
                    row[method + '_drop_ci_high_pp'] = 100 * f['ci_95']['high']
                else:
                    row[method + '_drop_ci_low_pp'] = row[method + '_drop_ci_high_pp'] = 0.0
            attacks.append(row)
        languages = []
        for language in LANGUAGES:
            rows = select(ledgers['D'], heldout_language=language)
            left, right = select(rows, method='lpcode_original'), select(rows, method='mstf')
            row = dict(language=language, **stats(left, right, ('seed',), {(s,) for s in SEEDS}), baseline_sample_sd=stdev(r['f1'] for r in left), method_sample_sd=stdev(r['f1'] for r in right))
            absolute(row, d['cell_summaries'][language], language)
            equal(row['baseline_sample_sd'], d['cell_summaries'][language]['lpcode_original']['f1_std'], language + ' baseline sample SD')
            equal(row['method_sample_sd'], d['cell_summaries'][language]['mstf']['f1_std'], language + ' method sample SD')
            f = d['paired_mstf_minus_lpcode']['by_heldout_language'][language]
            languages.append(interval(row, f['ci_95'], f['mean_delta_f1'], d_path))
        negatives = []
        n_path = 'results/negative_pair_robustness/summary.csv'
        frozen_negative = list(csv.DictReader(read(n_path).splitlines()))
        if len(frozen_negative) != 3: raise ValueError('negative summary coverage')
        for mode_name in ('current', 'random', 'hard'):
            rows = select(ledgers['negative'], model='xgb', negative_pair_mode=mode_name)
            row = dict(negative_mode=mode_name, **stats(select(rows, representation='concat'), select(rows, representation='concat_delta')))
            matches = [f for f in frozen_negative if f['negative_pairing'] == mode_name]
            if len(matches) != 1: raise ValueError('duplicate/missing negative summary')
            f = matches[0]
            equal(row['baseline'], float(f['baseline_f1_mean']), mode_name + ' baseline')
            equal(row['method'], float(f['mstf_f1_mean']), mode_name + ' method')
            negatives.append(interval(row, {'low': float(f['ci_95_low']), 'high': float(f['ci_95_high'])}, float(f['delta_f1_mean']), n_path))
        for number, rows in [(4, generators), (5, attacks), (6, languages), (7, negatives)]: save(f'table{number}', rows)
        for name, rows in [('main', main), ('variants', variants), ('contrasts', contrasts), ('transformations', attacks), ('generator', generators), ('language', languages), ('negative', negatives)]: save('figure_' + name, rows)
        m_path = 'results/05_mechanism_analysis/feature_importance/mechanism_summary.json'
        mechanism = read(m_path)
        grouped_path = 'results/05_mechanism_analysis/feature_importance/grouped_importance.csv'
        validate_mechanism(mechanism, list(csv.DictReader(read(grouped_path).splitlines())))
        groups = [{'environment': env, **row, 'source': m_path} for env, ranking in mechanism['rankings'].items() for row in ranking['group_rows']]
        save('figure_mechanism', groups)
    provenance['summary_validation'] = 'PASS: absolute means, paired differences, and sample SD' if mode == 'all-saved' else 'PASS: paired differences; sample SD not in requested table'
    write_json(output / 'provenance.json', provenance)
    return {'status': 'PASS', 'model_fitting': False, 'mode': mode, 'source_count': len(provenance['sources'])}

import zipfile, tempfile, os, subprocess, sys

zip_path = r'/mnt/c/Users/Karanpreet Singh/Downloads/2026MCS2250 (5).zip'
print('=' * 65)
print('RUNNING CI CHECKS ON 2026MCS2250 (5).ZIP IN LINUX WSL')
print('=' * 65)

with tempfile.TemporaryDirectory() as tmpdir:
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmpdir)
    
    extracted_root = os.path.join(tmpdir, '2026MCS2250')
    sub_dir = os.path.join(extracted_root, 'submission')
    
    print('\n[CI Step 1] Running: python3 setup.py build_ext --inplace')
    res = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=sub_dir, capture_output=True, text=True)
    print('Return Code:', res.returncode)
    print('STDOUT:\n', res.stdout.strip())
    print('STDERR:\n', res.stderr.strip())
    
    print('\n[CI Step 2] Running: pytest tests/ -v')
    res_pytest = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'], cwd=extracted_root, capture_output=True, text=True)
    print('Pytest Exit Code:', res_pytest.returncode)
    if res_pytest.returncode == 0:
        print('All pytest unit tests PASSED.')
    else:
        print('Pytest Output:\n', res_pytest.stdout)

    print('\n[CI Step 3] Running: Smoke Test on Toy Set')
    cmd = [
        sys.executable, '-m', 'harness.run_harness',
        '--corpus', 'data/toy/corpus.jsonl',
        '--queries', 'data/toy/queries_dev.tsv',
        '--qrels', 'data/toy/qrels_dev.txt',
        '--k', '10',
        '--submission', 'submission.retrieve'
    ]
    res_harness = subprocess.run(cmd, cwd=extracted_root, capture_output=True, text=True)
    print('Harness Exit Code:', res_harness.returncode)
    if res_harness.returncode == 0:
        print('Toy Smoke Test PASSED.')
    else:
        print('Harness Output:\n', res_harness.stderr)

print('=' * 65)

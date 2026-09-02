import zipfile, tempfile, os, subprocess, sys

zip_path = r'/mnt/e/IR Assignment/2026MCS2250.zip'
print('=' * 70)
print('STARTING VERIFICATION DIRECTLY ON: 2026MCS2250.ZIP')
print('=' * 70)

# Create an isolated sandbox directory
with tempfile.TemporaryDirectory() as sandbox:
    print(f'1. Unzipping {zip_path} into isolated sandbox: {sandbox} ...')
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(sandbox)
    
    extracted_root = os.path.join(sandbox, '2026MCS2250')
    print('   Successfully extracted. Contents:', os.listdir(extracted_root))
    
    # 1. Test Setup.py in submission/
    print('\n2. Testing Docker Build Step in Unzipped Folder...')
    sub_dir = os.path.join(extracted_root, 'submission')
    res_build = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=sub_dir, capture_output=True, text=True)
    if res_build.returncode == 0:
        print('   [PASSED] setup.py build_ext completed with Exit Code 0 (No Flat-Layout Error)!')
    else:
        print('   [FAILED] setup.py failed:', res_build.stderr)
        sys.exit(1)
        
    # 2. Test Conformance Pytest Tests
    print('\n3. Testing Conformance Unit Tests in Unzipped Folder...')
    res_pytest = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'], cwd=extracted_root, capture_output=True, text=True)
    if res_pytest.returncode == 0:
        print('   [PASSED] All 17 Conformance & Metric Tests PASSED!')
    else:
        print('   [FAILED] Pytest tests failed:\n', res_pytest.stdout)
        sys.exit(1)

    # 3. Test Full Harness Smoke Run
    print('\n4. Testing Full Harness Run in Unzipped Folder...')
    cmd_harness = [
        sys.executable, '-m', 'harness.run_harness',
        '--corpus', 'data/toy/corpus.jsonl',
        '--queries', 'data/toy/queries_dev.tsv',
        '--qrels', 'data/toy/qrels_dev.txt',
        '--k', '10',
        '--submission', 'submission.retrieve'
    ]
    res_harness = subprocess.run(cmd_harness, cwd=extracted_root, capture_output=True, text=True)
    if res_harness.returncode == 0:
        print('   [PASSED] Harness run on Toy Corpus completed with Exit Code 0!')
    else:
        print('   [FAILED] Harness run failed:\n', res_harness.stderr)
        sys.exit(1)

print('=' * 70)
print('FINAL RESULT FOR 2026MCS2250.ZIP: 100% VERIFIED & READY FOR SUBMISSION!')
print('=' * 70)

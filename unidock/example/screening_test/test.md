## Verification.py
-files : pdbqt files path \
-out : result json path \
example:
```shell
python verification.py -files "result/def" -out 'result/result.json'      
```
## diff.py
-f1: first result json \
-f2: second result json \
example:
```shell
python diff.py -f1 "result/result.json" -f2 "result/result1.json"  
```
1. Install the dependncies: pip install -r requirements_dev.txt

2. The list of documents:
2.1 "conf.py":  The configuration file
    - Inference mode: max, mean, min, default mean                                
    - Discard ratio: default 0.9
2.2 "attention_map.py": Generating attention map
2.3 Folder "input":  The store of input data
2.4 Folder "mlruns": The repository of mlflow output
2.5 Folder "vit_pytorch": The store of VIT architecture
              
3. Executing command: python attention_map.py 
                                                                    
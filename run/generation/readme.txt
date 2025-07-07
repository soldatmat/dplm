1 Fix PROJECT_ROOT bug:
    in `dplm/logs/<run_name>/.hydra/config.yaml` update line:
    ```yaml
    paths:
        root_dir: /home2/soldat/documents/dplm # ${oc.env:PROJECT_ROOT}
    ```

2 Call `sh generate_distributed.sh` to generate sequences with a trained DPLM model.
    - Change constants in the `generate_distributed.sh` file to specify the model, number of sequences etc.
    - The script will submit many short jobs to distribute the generation.
    - Pay attention to the temperature parameter, especially with a model trained up to ~0 loss.

3 Collect the generated sequence with `collect_sequences.py`.

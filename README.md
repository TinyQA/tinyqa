# The TinyQA Dataset

The content of dataset can be founded at https://huggingface.co/datasets/TinyQA/TinyQA.

To generate baseline results of your model, you can use this command:
```commandline
python -m generation_baseline --model your_model_name_here --file dataset_csv_here
```
The original version of generation_baseline requires huggingface transformers library, which is used in generate contents locally.

To generate the online-search assisted results, you can use this command:
```commandline
python -m search_generation_agent --model model_name_here --file search_group_csv_here --base_url api_base_url_here
```

You should set env ```OPENAI_API_KEY``` to call the api.

To generate the online-search and calculator assisted results, you can use this command:
```commandline
python -m agent_generation_agent --model model_name_here --file agent_group_csv_here --base_url api_base_url_here
```

You should set env ```OPENAI_API_KEY``` to call the api.

If you want to evaluate the models stored locally by huggingface transformers library, you can modify the script referencing ```generation_baseline.py```.

To evaluate the results, you can use this command:
```commandline
python -m scorer --model model_under_test_here --task the_task_here --scorer scorer_model_here --base_url api_base_url_here
```
v
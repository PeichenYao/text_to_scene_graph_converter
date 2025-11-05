Text2SceneGraph Converter is a tool designed to automatically generate scene graphs from narrative text. By using natural language processing techniques and dependency parsing, this project extracts objects, attributes, and relationships from short stories and converts them into structured scene graphs. These scene graphs are organized in a time-sequenced manner, making them useful for tasks such as multi-frame visualization, semantic search, and story understanding.

## Key Features

- **Automatic Scene Graph Generation:** Converts narrative text into scene graphs with nodes and edges representing entities and relationships.
- **Time-sequenced Organization:** Generates scene graphs across multiple frames, maintaining temporal coherence.
- **Extensive Customization:** Users can modify parsing and extraction parameters to fit their specific needs.
- **Evaluation Metrics:** Implements various evaluation techniques, including object/attribute/relationship matching and semantic consistency measures.
- **Scene Graph Enhancement:** Uses LLMs to enhance the generated scene graphs, ensuring richer details for better story representation.
- **Prompt Generation for Text-to-Image Tasks:** Transforms scene graphs into effective prompts for text-to-image generation.

## Installation

1. **Clone the repository**
	```bash
	git clone xxx
	cd text_to_scene_graph_converter
	```
2. **Create and activate virtual environment**
	```
	conda create -n env_name python==3.12
	conda activate env_name
	```
3. **Install dependencies**
	```bash
	pip install -r requirements.txt
	python -m spacy download en_core_web_trf
	```

## Configuration
The `config.py` file is used to define various configuration settings for the project. 

1. **Paths**
	- `TEXTS_FILE`: Path to the text data file.
	- `SCENE_GRAPHS_FILE`: Path to the scene graph data file.
	- `OUTPUT_PATH`: Directory where the output will be saved.
2. **API Configurations**
	- `OPENAI_API_KEY`: Your OpenAI API key for interacting with the OpenAI service.
	- `MODEL_NAME`: The model name you wish to use ( `gpt-5-mini` in default).
3. **Evaluation Settings**
	- `EVALUATION_METRIC`: The metric used to evaluate the performance (`SPICE` in default).
	- `EVALUATION_BATCH_SIZE`: The batch size to use during evaluation.
4. **Logging Settings**
	- `LOG_FILE`: The path where the logs will be saved.
	- `LOG_LEVEL`: The level of logging (`INFO`, `DEBUG`, `ERROR`, `INFO` in default).
5. **Proxy Settings**
	- `USE_PROXY`: Set to `True` if you need to use a proxy (`False` in default).
	- `PROXY_URL`: The URL of the proxy server.

## Usage

### Generate your dataset
Our project provides a method to generate usable short story texts and corresponding scene graph datasets via LLM, which can be generated in the following way.

**Notice**:
- To use this feature, you need to first configure the `Paths` and `API Configurations` sections in `config.py`.
- Specify the number of generated items by using `--num_data {int_of_number}`.

```bash
python data.py --num_data 5
```

### Inference

Our project offers multiple modes.

**Notice**:
- To use the feature with LLM, you need to first configure the `Paths` and `API Configurations` sections in `config.py`.

1. Text to scene graph
	```bash
	python evaluate.py
	```
2. Text to scene graph + 
	- Evaluation
		```bash
		python evaluate.py --evaluate
		```
	- Scene graph enhancement
		```bash
		python evaluate.py --enrich --enrich-path=path_to_save_results
		```
	- Scene graph to text-to-image prompts
		```bash
		python evaluate.py --prompt --prompt-path=path_to_save_results
		```

# Chat99

Chat99 is an intelligent AI assistant with advanced memory, multi-model capabilities, and dynamic routing using RouteLLM.

Chat99 is a pre-curser to the intended end goal of creating Agent99 and an agentic framework that dynamically routes between local/lower-end models for simpler or specific task execution while also dynamically assigning model parameters to achieve the best results for each specific task. 

## Features

- Dynamic model selection using RouteLLM
- Advanced memory management for context retention
- Support for multiple AI models (Claude, GPT-4, etc.)
- Calibration tool for optimizing model selection thresholds
- Local model support using Ollama

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/GaryOcean428/Agent99.git
   cd Agent99
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables by copying the `.env.template` file to `.env` and filling in your API keys:
   ```
   cp .env.template .env
   ```
   Then edit the `.env` file with your actual API keys.

## Usage

1. Run the calibration script to find the optimal threshold for model selection:
   ```
   python calibrate_threshold.py --sample-queries sample_queries.json --router mf --strong-model-pct 0.5
   ```

2. Start the Chat99 assistant:
   ```
   python main.py --use-dynamic-routing --router mf --threshold <threshold_from_calibration>
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

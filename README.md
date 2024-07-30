# Chat99

Chat99 is an intelligent AI assistant with advanced memory, multi-model capabilities, and dynamic routing using RouteLLM.

# Agent99

Chat99 is a pre-curser to the intended end goal of creating Agent99 and an agentic framework that dynamically routes between local/lower-end models for simpler or specific task execution while also assigning model parameters to achieve the best results for each specific task.

Agent99, as a project name, is a loose call back to the 60's comedy show "Get Smart". Get Smart was and probably still is on re-runs every afternoon, and I'd watch it after school. And like many of you... Agent 99 was of particular interest. Although that wasn't part of why I named it that. It was the first thing that popped into my head when I thought, "Agent... Agent... what?"

# Scroll to Chat99 Readme, Agent99 still under Development. 

## Features

- Dynamic model selection using RouteLLM
- Advanced memory management for context retention
- Support for multiple AI models (Claude, GPT-4, etc.)
- Calibration tool for optimizing model selection thresholds
- Local model support using Ollama

### Dev notes

- memory_manager.py is a temporary solution so that focus can be placed on other project areas. 

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

# Chat99

Chat99 is an intelligent AI assistant with advanced memory, multi-model capabilities, and dynamic routing using AdvancedRouter.

## Features

- Dynamic model selection using AdvancedRouter
- Advanced memory management for context retention
- Support for multiple AI models (Claude, Llama, etc.)
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
   python calibrate_threshold.py --sample-queries sample_queries.json --strong-model-pct 0.5
   ```

2. Start the Chat99 assistant:
   ```
   python chat99.py
   ```

3. Interact with the assistant by typing your messages when prompted.

4. To exit the program, you can:
   - Type "exit" when prompted for input
   - Use the keyboard interrupt (Ctrl + C on most systems, Cmd + C on macOS)

## Project Structure

- `chat99.py`: Main script for the chat interface
- `advanced_router.py`: Handles dynamic model selection
- `config.py`: Configuration settings for the project
- `memory_manager.py`: Manages conversation context and memory
- `models.py`: Defines available AI models
- `calibrate_threshold.py`: Script for calibrating the routing threshold

## Current Status

- Functional chat interface with dynamic model routing
- Support for multiple AI models (Claude 3.5, Llama 3.1 70B, Llama 3.1 8B)
- Basic complexity assessment for query routing
- Calibration script for optimizing routing thresholds

## Future Development

- Enhance complexity assessment with more sophisticated NLP techniques
- Implement dynamic allocation of temperature and top-p parameters
- Develop multi-dimensional routing considering both model and parameter selection
- Integrate user preferences for response style
- Implement adaptive learning for continuous improvement of routing decisions
- Explore multi-model ensemble responses for comprehensive answers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Chat With Paper

## Overview

Chat With Paper is a Python-based tool that utilizes Language Models (LLMs) and OpenAI APIs to enable interactive discussions with scientific papers. This repository provides a simple script to initiate conversations with research papers, allowing users to engage in a chat-like format to better understand and discuss the content.

## Features

- **Conversational Interface:** Interact with scientific papers using natural language.
- **Language Models:** Utilizes powerful language models for enhanced conversational experiences.
- **PDF Support:** Currently supports PDF files for input.

## Installation

To get started, ensure you have Python installed on your system. Clone this repository and navigate to the project directory in your terminal. Then, run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

This command installs the necessary Python packages to run the script successfully.

## OpenAI API Key Setup

This repository uses OpenAI APIs for enhanced language processing. To use the OpenAI API, you need to obtain an API key. Follow these steps:

1. **Get OpenAI API Key:**
   - Visit the [OpenAI website](https://beta.openai.com/signup/).
   - Sign up or log in to your OpenAI account.
   - Follow the instructions to get your API key.

2. **Save API Key in .env File:**
   - Create a file named `.env` in the project directory.
   - Open the `.env` file in a text editor and add the following line:
     ```env
     OPENAI_API_KEY=your_api_key_here
     ```
     Replace `your_api_key_here` with your actual OpenAI API key.

   **Note:** The `.env` file is added to your `.gitignore` to prevent accidentally sharing your API key. But it is always safer to check it again.

## Usage

After installing the requirements and setting up your OpenAI API key, you can use the script to chat with a scientific paper. Follow the steps below:

1. **Navigate to the project directory:**
   ```bash
   cd path/to/chat-with-paper
   ```

2. **Run the script with a PDF file:**
   ```bash
   python main.py path_to/paper.pdf
   ```

   Replace `path_to/paper.pdf` with the actual path to the PDF file you want to discuss.

## Example

```bash
python main.py example_paper.pdf
```

## Contributing

If you have ideas for improvements or encounter any issues, feel free to open an issue or submit a pull request. Your contributions are welcome!

---

Happy chatting with scientific papers! 📚💬

Note: README.md mostly generated by ChatGPT-3.5

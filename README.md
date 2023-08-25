# LangChain-Ecommerce-Chatbot

AI-powered E-Commerce Assistant Chatbot that helps customers with their day-to-day seacrhing & purchase. It's developed using OpenAI's `gpt-3.5-turbo`` model, LangChain & Qdrant as a Vector Store.


## Prerequisites

Before using this app boilerplate, make sure you have the following:

- Python 3.x installed on your system.
- An active Qdrant Managed Cloud Service account.
- Your Qdrant Managed Cloud Service credentials.

## Installation

To install and run the app boilerplate, follow these steps:

1. Clone this repository to your local machine:

```shell
git clone https://github.com/MoRaouf/LangChain-Ecommerce-Chatbot.git
```

2. Change into the project directory:

```
cd LangChain-Ecommerce-Chatbot
```

3. Create a virtualenv and activate it

```
python3 -m venv .venv && source .venv/bin/activate
```

4. Install the required Python packages using pip:

```
pip install -r requirements.txt
```

## Configuration

To configure the app boilerplate to work with your Qdrant Managed Cloud Service account, you need to add your credentials to the `.env` file. Follow these steps:

1. Copy the `.env.example` file and rename it to `.env`:

```
cp .env.example .env
```

2. Open the `.env` file in a text editor and provide your Qdrant Managed Cloud Service and OpenAI   credentials:

```
OPENAI_API_KEY =           
LANGCHAIN_API_KEY = 
QDRANT_API_KEY = 
```

## Usage

Once you have completed the installation and configuration steps, you can run the app using the following command:

```
streamlit run app.py
```
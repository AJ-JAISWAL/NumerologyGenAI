# NumerologyGenAI(Tantralogy)

AyurParvani is a powerful tool designed to provide Numerology_ With Tantra, Ayurveda, and Astrology information by answering user queries using state-of-the-art language models and vector stores. This README will guide you through the setup and usage of the Tantralogy.

## Table of Contents

- [Introduction](#NumerologyGenAI(Tantralogy))
- [Table of Contents](#table-of-contents)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Usage](#usage)

## Prerequisites

Before you can start using the Tantralogy, make sure you have the following prerequisites installed on your system:

- Python 3.12 or higher
- Required Python packages (you can install them using pip):
    - langchain
    - Streamlit
    - sentence-transformers
    - Pymongo(for storing data as database also known as mongodb)
    - PyPDF2 (for PDF document loading)

## Installation

1. Clone this repository to your local machine.

    ```bash
    [git clone https://github.com/AJ-JAISWAL/NumerologyGenAI.git]
    cd Tantralogy
    ```

2. Create a Python virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

3. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Download the required language models and data. Please refer to the Langchain documentation for specific instructions on how to download and set up the language model and vector store.

5. Set up the pymongo environment in your project so that the vector embedding and content fetched as per your needs.

## Getting Started

To get started with the Tantralogy, you need to:

1. Set up your environment and install the required packages as described in the Installation section.

2. Configure your project with mongo atlas cloud.

3. Prepare the language model and data as per the Langchain documentation.

4. Start the application by running the provided Python script or integrating it into your application.

## Usage

The Tantralogy can be used for answering Numerology_ With Tantra, Ayurveda, and Astrology related queries. To use the application, you can follow these steps:

1. Start the application by running your application or using the provided Python script.

2. Send a astrology-related or numerology query to the application.

3. The bot will provide a response based on the information available in its database.

4. If sources are found, they will be provided alongside the answer.

5. The application can be customized to return specific information based on the query and context provided.

![Screenshot 2024-12-01 203259](https://github.com/user-attachments/assets/0492d3f9-37a0-4877-aca2-1b35f7a22f36)
![Screenshot 2024-12-01 203337](https://github.com/user-attachments/assets/597bcf02-f05b-4c25-8003-b6d4e4770563)

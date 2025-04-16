# Retrieval Based Ticker Information Bot
### This project is a self-learning project, almost all the process done in the project is very basic and beginner level. 

### The goal of this project is to have built a RAG based system by the end. In order to achieve this I am constantly learning and trying new things. 

## Step 1 : Collection of data
### The data is collected from 2 public kaggle datasets : <br> <ul> <li> Financial news :<url>https://www.kaggle.com/datasets/notlucasp/financial-news-headlines?select=guardian_headlines.csv<url> <li> Stock Information : <url>https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset <url> </ul> <br> The data is collected, read, cleaned and pre-processed.

## Step 2: Storing the data
### Once the data is deemed fit to be used, the data is first embed using a low-cost light-weight model <q>text-embedding-3-small</q>. The reason for embedding is because the data is going to be stored in a vector database where only the embeddings/vectors are stored. <br> <br> We are using Pinecone for this project, the final data will be upsert into that. Before beginnning with the code kindly go through the steps to create a vector database and create an index where your data will be stored. 

## Step 3: Collecting query
### Now that the context for the model is saved, we are ready to get the user's prompt/question. This question will be obtained through a chat interface. We will create embeddings of the query as well and then use that to search the vector database for the nearest matching entries. These entries will be retrieved from the database for the context that will be passed to the AI model. 

## Step 4: Prompt engineering
### With the context, query ready we simply create a prompt that teaches the model how to use the context and answwer the query respectively. We will be tweaking the temperature for this process so that the model is more dependent on the context that was provided, reducing hallucintations but before getting to this step all the data has to be verified and it should be enough for a model to come up with a solution. 
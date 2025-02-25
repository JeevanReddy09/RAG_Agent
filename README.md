# RAG_Agent using Langchain

This repository demonstrates a flexible question answering agent built with LangChain. The project processes PDF documents, builds vector stores, and uses retrieval-based QA to answer user queries. It is designed to work with Groq and OpenAI models—and can be easily adapted to use open source language models.

Features: 

PDF Processing:
Extracts and splits text from PDF documents using PyPDFLoader and RecursiveCharacterTextSplitter.

Vector Store Integration:
Creates vector stores with both Chroma and Pinecone for efficient document retrieval.

Retrieval-Based Question Answering:
Uses LangChain’s RetrievalQA chain to find and answer questions based on relevant document sections.

Custom Agent with ReAct:
Implements a Zero-Shot ReAct agent with custom prompt templates and output parsers for dynamic reasoning.

Model Flexibility:
Supports Groq and OpenAI models by default and is easily configurable to use open source models.

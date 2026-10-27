# RAG Performance Monitoring Specification
 
## Objective
 
Add a lightweight performance monitoring dashboard to the existing RAG/document chunking application.
 
The dashboard must display:
 
- Response time
- CPU usage
- RAM/memory usage
- Number of chunks retrieved
- Timing for individual RAG pipeline stages
 
The monitoring must measure the **backend RAG process**, not only browser performance.
 
---
 
# 1. Existing RAG Pipeline
 
The current system follows this general flow:
 
```text
User Upload Document
      |
      v
Generate Chunk 
      |
      v     
User Question
      |
      v
Generate Query Embedding
      |
      v
MongoDB Vector Search
      |
      v
Retrieve Relevant Chunks
      |
      v
Generate RAG Response
      |
      v
Return Response to Browser

Add more which i didnt mention on the pipeline but the system have those functionalities , 
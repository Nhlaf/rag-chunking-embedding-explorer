from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate

OUTPUT = 'rag_app_explanation.pdf'

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='CustomTitle', parent=styles['Title'], alignment=1, fontSize=22, leading=28, spaceAfter=18))
styles.add(ParagraphStyle(name='CustomHeading', parent=styles['Heading2'], fontSize=14, leading=18, spaceAfter=8, spaceBefore=16))
styles.add(ParagraphStyle(name='CustomBody', parent=styles['BodyText'], fontSize=10.5, leading=15, spaceAfter=6))
styles.add(ParagraphStyle(name='CustomBullet', parent=styles['BodyText'], fontSize=10.5, leftIndent=18, bulletIndent=12, spaceAfter=5))

story = []
story.append(Paragraph('RAG Application Overview', styles['CustomTitle']))
story.append(Paragraph('How the whole application works', styles['CustomHeading']))
story.append(Paragraph(
    'This project is a small RAG (Retrieval-Augmented Generation) explorer: it takes a document, turns it into text, splits it into chunks, creates embeddings, stores those vectors, retrieves the most relevant chunks for a user question, and optionally uses an LLM to answer from those chunks.',
    styles['CustomBody'],
))
story.append(Paragraph('The overall flow is:', styles['CustomBody']))
story.append(ListFlowable([
    ListItem(Paragraph('Upload a document or paste text', styles['CustomBullet'])),
    ListItem(Paragraph('Parse the file into readable text', styles['CustomBullet'])),
    ListItem(Paragraph('Classify whether it is structured or narrative', styles['CustomBullet'])),
    ListItem(Paragraph('Chunk the content into smaller, searchable pieces', styles['CustomBullet'])),
    ListItem(Paragraph('Generate an embedding vector for each chunk', styles['CustomBullet'])),
    ListItem(Paragraph('Index the vectors in memory', styles['CustomBullet'])),
    ListItem(Paragraph('Embed the user query', styles['CustomBullet'])),
    ListItem(Paragraph('Search for the most similar chunks', styles['CustomBullet'])),
    ListItem(Paragraph('Check relevance, then generate a grounded answer from retrieved chunks', styles['CustomBullet'])),
], bulletType='bullet', bulletFontName='Helvetica', bulletFontSize=9, leftIndent=18, spacing='normal'))

story.append(Paragraph('Main flow in plain English', styles['CustomHeading']))
story.append(Paragraph('1) Input and parsing', styles['CustomHeading']))
story.append(Paragraph('The app accepts PDF, DOCX, XLSX, and XLS files, or raw text pasted manually. The parser extracts text from the file, converts spreadsheet rows into readable records, and decides whether the document is structured or unstructured.', styles['CustomBody']))
story.append(Paragraph('2) Chunking', styles['CustomHeading']))
story.append(Paragraph('The chunking service breaks long content into smaller pieces. Structured data gets row-based or section-based chunks, while narrative content gets heading or paragraph-aware chunks. This helps the embedding model work on focused segments instead of giant raw text blocks.', styles['CustomBody']))
story.append(Paragraph('3) Embeddings', styles['CustomHeading']))
story.append(Paragraph('The embedding logic uses SentenceTransformers with all-MiniLM-L6-v2. Each chunk is converted into a numeric vector. The same model also embeds the user query so the query vector can be compared with chunk vectors semantically.', styles['CustomBody']))
story.append(Paragraph('4) Vector storage and search', styles['CustomHeading']))
story.append(Paragraph('The in-memory vector index stores the chunk id, chunk text, and vector. Search supports cosine similarity, dot product, or Euclidean distance. The search endpoint embeds the query, finds matching chunks, ranks them, and returns the top-ranked results.', styles['CustomBody']))
story.append(Paragraph('5) Relevance and answer generation', styles['CustomHeading']))
story.append(Paragraph('The LLM logic checks whether the retrieved chunks are close enough to answer the question and then calls Groq with the retrieved chunk context. If the similarity score is weak, it refuses instead of hallucinating.', styles['CustomBody']))
story.append(Paragraph('6) Persistence', styles['CustomHeading']))
story.append(Paragraph('Optional MongoDB storage is used when MONGODB_URI is configured. If it is not configured, the app falls back to in-memory storage.', styles['CustomBody']))
story.append(Paragraph('Brief explanation of the main file', styles['CustomHeading']))
story.append(Paragraph('The main orchestrator is backend/main.py. It creates the FastAPI app, defines the API routes, keeps the in-memory document state and vector store, and records performance metrics such as CPU, memory, and timing per stage.', styles['CustomBody']))
story.append(Paragraph('In other words, backend/main.py is the glue layer connecting parsing, chunking, embedding, indexing, retrieval, and answer generation.', styles['CustomBody']))
story.append(Paragraph('Frontend role', styles['CustomHeading']))
story.append(Paragraph('The UI in the frontend files uploads files, calls the backend endpoints, shows extracted text, displays chunk statistics, renders embeddings, sends queries, and displays the answer with supporting chunks. This makes the whole retrieval pipeline visible to the user.', styles['CustomBody']))
story.append(Paragraph('In one sentence', styles['CustomHeading']))
story.append(Paragraph('This application is a beginner-friendly RAG pipeline demo: it turns files into text, chunks and embeds them, searches semantically, and answers questions using only the retrieved context.', styles['CustomBody']))

doc = SimpleDocTemplate(OUTPUT, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
doc.build(story)
print(f'Created {OUTPUT}')

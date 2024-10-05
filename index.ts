import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { pull } from "langchain/hub";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import * as dotenv from 'dotenv';
dotenv.config();

import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";

const loader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/"
);

const docs = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const splits = await textSplitter.splitDocuments(docs);
const vectorStore = await MemoryVectorStore.fromDocuments(
  splits,
  new OllamaEmbeddings()
);

// Retrieve and generate using the relevant snippets of the blog.
const retriever = vectorStore.asRetriever();
const prompt = await pull<ChatPromptTemplate>("rlm/rag-prompt");
const llm = new ChatOllama({
  model: "llama3.1",
  temperature: 0,
});

const ragChain = await createStuffDocumentsChain({
  llm,
  prompt,
  outputParser: new StringOutputParser(),
});

const retrievedDocs = await retriever.invoke("what is task decomposition");

const response = await ragChain.invoke({
  question: "What is task decomposition?",
  context: retrievedDocs,
});

let outputString = '';
for await (const output of response) {
  outputString += output;
}

console.log(outputString.trim());

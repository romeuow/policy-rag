# policy-rag

Assistente RAG (Retrieval-Augmented Generation) full-stack que responde perguntas sobre políticas internas
com citação do trecho e do documento de origem, para que colaboradores encontrem regras em segundos e o RH
pare de responder as mesmas dúvidas repetidamente.

![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)
![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Frontend](https://img.shields.io/badge/frontend-React%20%2B%20Vite%20%2B%20TS-61DAFB?logo=react&logoColor=black)

## Caso de uso real

Em uma consultoria de sustentabilidade, as regras do dia a dia (férias, reembolso, segurança da informação,
home office, conduta, benefícios, viagens) estavam espalhadas por dezenas de políticas em PDF e Markdown.
Colaboradores perdiam tempo procurando a cláusula certa, acabavam perguntando no chat corporativo e o RH
respondia as mesmas perguntas várias vezes por semana, muitas vezes com versões desatualizadas do documento.

A solução foi um assistente de perguntas e respostas sobre a base de políticas. Cada resposta vem acompanhada
das citações `[n]` que apontam para o documento, a seção e o trecho usados, de modo que a pessoa consegue
conferir a fonte e o RH consegue auditar o que o assistente disse. O RH ganhou um painel para atualizar
políticas (com reindexação idempotente), cadastrar FAQs que respondem antes do RAG e revisar respostas com
avaliação negativa, fechando o ciclo de melhoria contínua.

O que mudou: as perguntas repetitivas deixaram de chegar ao RH, as respostas passaram a citar a versão
vigente do documento e o time ganhou visibilidade sobre o que os colaboradores não encontram (métricas de
"sem resposta" e feedback negativo), o que orientou a revisão das próprias políticas.

Este repositório é uma **reimplementação genérica** do padrão aplicado em produção, com dados 100%
sintéticos: as políticas em `corpus/` são fictícias, escritas para o projeto, de uma empresa inventada
("Exemplo Consultoria"). Nenhum documento, prompt, nome ou número real foi utilizado.

## O que este projeto demonstra

- **Pipeline RAG completo**: parsing (Markdown/TXT, PDF opcional), chunking por heading com overlap,
  embeddings, busca híbrida (vetorial + BM25) com Reciprocal Rank Fusion, geração com citações.
- **Ingestão idempotente**: `chunk_id` derivado do hash do conteúdo; reingestão só escreve o que mudou e
  remove chunks órfãos.
- **Ports & adapters**: `EmbeddingProvider`, `VectorStore` e `LLMClient` são `Protocol`s com
  implementações de produção (Voyage AI, pgvector, Claude) e fakes determinísticos para testes e demo.
- **Streaming SSE** de ponta a ponta (`token` -> `citations` -> `done`), com parser incremental no
  frontend e codec testado no backend.
- **Guardrails**: limite de tamanho da pergunta, detecção de prompt injection nos chunks recuperados
  (flag em log e exclusão), resposta "não encontrei" quando a relevância fica abaixo do limiar, tratamento
  de `stop_reason == "refusal"`.
- **Segurança na borda**: rotas administrativas protegidas por `X-Admin-Token` com comparação em tempo
  constante; validação de entrada com Pydantic; segredos só via ambiente.
- **Observabilidade básica**: logging JSON estruturado com `extra` por evento e `GET /metrics` com
  contadores em memória.
- **Qualidade mensurável**: teste de recall@3 sobre um conjunto de perguntas versionado
  (`tests/fixtures/questions.yaml`), 79 testes offline, 99% de cobertura.
- **Operação**: CLI (`policy-rag ingest|serve`), Dockerfile multi-stage sem `uv` na imagem final e usuário
  não-root, `docker-compose` com `pgvector/pgvector:pg16` e migrations Alembic, CI com ruff + pytest + vitest.

## Arquitetura

```mermaid
flowchart LR
    subgraph Web["web/ (React + Vite)"]
        Chat["/ Chat (SSE)"]
        Admin["/admin (token)"]
    end

    subgraph API["FastAPI (src/policy_rag/api)"]
        ChatR["POST /chat"]
        FbR["POST /feedback"]
        AdmR["/admin/* (X-Admin-Token)"]
        Meta["GET /health, /metrics"]
    end

    subgraph RAG["RagPipeline"]
        FAQ["FaqMatcher"]
        Retr["HybridRetriever<br/>vetorial + BM25 -> RRF"]
        Guard["Guardrails<br/>injection screen / limiar"]
        LLM["LLMClient"]
    end

    subgraph Ports["Adapters"]
        Emb["EmbeddingProvider<br/>Fake | Voyage"]
        Store["VectorStore<br/>InMemory | PgVector"]
        Claude["AnthropicLLMClient<br/>messages.stream"]
        FakeLLM["FakeLLMClient<br/>extractivo"]
    end

    Ingest["policy-rag ingest corpus/<br/>parse -> chunk -> embed -> upsert"]

    Chat --> ChatR --> FAQ
    FAQ -- "sem match" --> Retr --> Guard --> LLM
    FAQ -- "match >= limiar" --> ChatR
    Retr --> Emb
    Retr --> Store
    LLM --> Claude
    LLM --> FakeLLM
    Admin --> AdmR --> Ingest --> Store
    Chat --> FbR
    Ingest --> Emb
```

### Componentes

| Camada | Módulos | Responsabilidade |
|---|---|---|
| Ingestão | `ingest/parsers.py`, `ingest/chunker.py` | Front matter, título, versão; divisão por headings (`##`, `###`) e por parágrafos com overlap; `chunk_id = sha256(doc_id, seção, texto)[:24]`. |
| Embeddings | `embeddings/fake.py`, `embeddings/voyage.py` | `FakeEmbeddingProvider`: feature hashing de unigramas e bigramas (tokenizer pt-BR com stopwords e stemming leve) em 256 dims normalizadas. `VoyageEmbeddingProvider`: `/embeddings` via httpx. |
| Store | `store/memory.py`, `store/pgvector.py`, `store/fusion.py` | Cosine + BM25 em memória; pgvector com `<=>`, índice HNSW e `tsvector` em português; RRF. |
| RAG | `rag/retriever.py`, `rag/faq.py`, `rag/guardrails.py`, `rag/pipeline.py`, `rag/prompts.py` | Busca híbrida, FAQ antes do RAG, triagem de injection, limiar de relevância, montagem de citações e eventos de stream. |
| LLM | `llm/anthropic_client.py`, `llm/fake.py` | `client.messages.stream(...)` iterando `text_stream`; verificação de `stop_reason`; fake extractivo determinístico. |
| Serviços | `services/*` | Registro de documentos e reindexação, histórico curto por sessão, métricas, feedback, FAQs. |
| API | `api/*` | Rotas, SSE, auth admin, schemas, container de dependências por `APP_MODE`, serving do `web/dist`. |

### Fluxo de uma requisição `POST /chat`

1. O body `{session_id, question}` é validado (Pydantic) e a pergunta passa por `validate_question`
   (normalização de espaços e limite `MAX_QUESTION_CHARS`, 413 se exceder).
2. `FaqMatcher` compara a pergunta com as FAQs cadastradas (max entre cosseno dos embeddings e Jaccard de
   tokens). Se `>= FAQ_MATCH_THRESHOLD`, a resposta da FAQ é emitida e o `done` sai com `source: "faq"`.
3. `HybridRetriever` embeda a pergunta, consulta top-k vetorial e top-k BM25 (filtrando documentos
   inativos) e funde os rankings com RRF (`k=60`).
4. `screen_chunks` descarta chunks com padrões de prompt injection e registra `prompt_injection_suspected`
   no log. Se não sobrar nada ou a relevância (`max(cosseno, bm25/(bm25+6))`) ficar abaixo de
   `MIN_RELEVANCE_SCORE`, responde "não encontrei" com `source: "none"`.
5. `LLMClient.stream_answer` recebe pergunta, chunks numerados e histórico da sessão; cada fragmento vira um
   evento SSE `token`. Em `refusal`, a resposta vira "não encontrei".
6. Emite `citations` (`doc_title`, `section`, `chunk_id`, `score`, `excerpt`) e `done`
   (`message_id`, `source`). O `message_id` é o que o frontend envia em `POST /feedback`.

### Decisões técnicas

| Decisão | Alternativa considerada | Por quê |
|---|---|---|
| Embeddings via Voyage AI | Embeddings da própria Claude API | A Claude API não oferece endpoint de embeddings; Voyage é o provedor recomendado pela Anthropic e tem modelos bons para recuperação multilíngue. A interface `EmbeddingProvider` isola a escolha. |
| Fake de embeddings por feature hashing de n-gramas | Mock que retorna vetores aleatórios | Vetores aleatórios não permitem testar recall. O hashing determinístico dá similaridade proporcional à sobreposição lexical, suficiente para `recall@3 = 1.0` no corpus e para validar o FAQ matcher offline. |
| Busca híbrida com RRF | Só vetorial | BM25 captura termos exatos ("abono pecuniário", "R$ 120,00") que embeddings podem diluir; RRF funde rankings sem precisar calibrar pesos entre escalas diferentes. |
| Limiar de "não encontrei" sobre scores brutos (cosseno e BM25 comprimido) | Limiar sobre o score RRF | O score RRF depende só da posição no ranking e é alto mesmo quando tudo é irrelevante; os scores brutos refletem semelhança real. |
| `chunk_id` por hash de conteúdo | ID sequencial / UUID | Torna a reingestão idempotente: chunks iguais são re-upsertados, só os alterados mudam de ID, órfãos são removidos. |
| Chunking por heading com overlap | Janela fixa de tokens | Políticas têm estrutura clara por seção; o chunk carrega o caminho de headings, o que melhora recuperação e deixa a citação legível ("2. Regras > 2.1 Regra A"). |
| SSE via `StreamingResponse` + codec próprio | WebSockets / `sse-starlette` | SSE é unidirecional e basta; o codec de 40 linhas é testado e evita dependência. |
| `messages.stream` + `text_stream` | `messages.create` sem stream | Respostas longas chegam incrementalmente ao usuário; `get_final_message()` dá acesso ao `stop_reason` ao final. |
| Estado em memória (sessões, feedback, FAQs, métricas) | Postgres/Redis | Mantém o demo sem dependências e deixa clara a fronteira: só os chunks têm store de produção. Persistir o resto é item de roadmap. |
| Admin via `X-Admin-Token` + `secrets.compare_digest` | OAuth/JWT | Suficiente para um painel interno de RH atrás de VPN; comparação em tempo constante evita timing attacks. |
| API serve `web/dist` com fallback de SPA | Nginx separado | Um único container atende tudo em demo; rotas da API têm precedência e caminhos são confinados ao `dist`. |

## Como executar

### Pré-requisitos

- Python 3.12+ e [`uv`](https://docs.astral.sh/uv/)
- Node 24+ e npm (apenas para o frontend)
- Docker (opcional, para `make up`)

### Instalação

```bash
make install            # uv sync --all-groups && cd web && npm install
cp .env.example .env    # revise ADMIN_TOKEN; em demo nada mais é necessário
```

### Modo demo (sem credenciais externas)

`APP_MODE=demo` (padrão) usa `FakeEmbeddingProvider`, `InMemoryVectorStore` e `FakeLLMClient`. O corpus em
`CORPUS_DIR` é ingerido automaticamente quando a API sobe.

```bash
uv run policy-rag ingest corpus/     # mostra documentos e chunks gerados (64 chunks em 7 documentos)
make dev                             # API em http://localhost:8000
cd web && npm run dev                # frontend em http://localhost:5173 (proxy para a API)
```

Para servir o frontend pela própria API: `cd web && npm run build` e reinicie `make dev`; a página fica em
`http://localhost:8000/` e o admin em `http://localhost:8000/admin`.

### Modo produção

```bash
APP_MODE=prod
ANTHROPIC_API_KEY=...          # lido pelo SDK anthropic
LLM_MODEL=claude-opus-5-5
VOYAGE_API_KEY=...
EMBEDDING_DIM=1024             # dimensão do modelo Voyage escolhido
DATABASE_URL=postgresql://policy:policy@localhost:5432/policy_rag
```

```bash
uv run alembic upgrade head          # cria a tabela chunks, índice HNSW e índice GIN
uv run policy-rag ingest corpus/     # persiste em pgvector
uv run policy-rag serve
```

### Docker

```bash
make up      # postgres (pgvector/pgvector:pg16) -> migrate (alembic upgrade head) -> api (:8000)
make down
```

O serviço `api` lê `.env`; por padrão sobe em `APP_MODE=demo` mesmo com o Postgres disponível. Troque para
`prod` e informe as chaves para usar Voyage, pgvector e Claude.

### Exemplos de requisição

Health:

```bash
curl -s localhost:8000/health
# {"status":"ok","version":"0.1.0","mode":"demo","documents":7,"chunks":64}
```

Chat com streaming SSE:

```bash
curl -N -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo-1","question":"Qual o limite diário de alimentação em viagem nacional?"}'
```

```text
event: token
data: {"text": "Com "}

event: token
data: {"text": "base "}

...

event: token
data: {"text": "R$ 120,00 (cento e vinte reais), dividido em café da manhã ... [1]"}

event: citations
data: [{"index": 1, "doc_id": "02-politica-de-reembolso", "doc_title": "Política de Reembolso de Despesas",
        "section": "4. Limites por categoria > 4.1 Alimentação em viagem", "chunk_id": "9c0e…",
        "score": 0.03252, "excerpt": "O limite diário de alimentação em viagem nacional é de R$ 120,00 …"}, ...]

event: done
data: {"message_id": "5f1c…", "source": "rag", "session_id": "demo-1"}
```

Feedback (usa o `message_id` do evento `done`):

```bash
curl -s -X POST localhost:8000/feedback -H 'Content-Type: application/json' \
  -d '{"message_id":"5f1c…","rating":"down","comment":"faltou o valor internacional"}'
# 201 {"feedback_id":"…","message_id":"5f1c…","rating":"down","comment":"faltou o valor internacional", ...}
```

Admin (sem token responde 401):

```bash
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/admin/documents      # 401
curl -s -H 'X-Admin-Token: change-me-admin-token' localhost:8000/admin/documents | jq '.[0]'
curl -s -X POST -H 'X-Admin-Token: change-me-admin-token' -H 'Content-Type: application/json' \
  localhost:8000/admin/faqs -d '{"question":"Qual o telefone do RH?","answer":"11 90000-0000"}'
curl -s -X POST -H 'X-Admin-Token: change-me-admin-token' \
  localhost:8000/admin/documents/01-politica-de-ferias/reindex
curl -s -H 'X-Admin-Token: change-me-admin-token' 'localhost:8000/admin/feedback?rating=down'
```

Métricas:

```bash
curl -s localhost:8000/metrics
# {"questions_total":3,"faq_hits_total":1,"no_answer_total":0,"rag_answers_total":2,
#  "feedback_up_total":0,"feedback_down_total":1,"injection_flags_total":0,"errors_total":0}
```

## Testes

```bash
make test                      # pytest com cobertura + vitest
uv run pytest -q --cov=src     # só backend
cd web && npm test             # só frontend
```

Todos os testes rodam **offline**, sem chaves e sem rede: os fixtures em `tests/conftest.py` montam o container em
`APP_MODE=demo` com os fakes.

| Tipo | Arquivos | O que cobre |
|---|---|---|
| Unit | `tests/unit/test_chunker.py`, `test_parsers.py` | Divisão por headings, limites e overlap, determinismo dos `chunk_id`, front matter, PDF via stub. |
| Unit | `test_embeddings.py` | Determinismo e normalização do fake; `VoyageEmbeddingProvider` com `httpx.MockTransport` (ordenação por índice, normalização, headers). |
| Unit | `test_fusion.py`, `test_memory_store.py` | Fórmula do RRF, ordenação, limite; upsert idempotente, BM25, cosine, filtro por documento. |
| Unit | `test_faq.py`, `test_guardrails.py` | Match exato e paráfrase, limiar; validação de pergunta, padrões de injection, triagem de chunks. |
| Unit | `test_sse.py`, `test_llm.py` | Codec SSE (comentários, `data` multilinha, CRLF); fake extractivo; `AnthropicLLMClient` com stub via monkeypatch (prompt montado, `text_stream`, `refusal` -> erro, `max_tokens` -> warning). |
| Unit | `test_services.py`, `test_documents.py`, `test_container.py`, `test_cli.py` | Formatter JSON, sessões, métricas, stores; registry/reindex/stale chunks; wiring por `APP_MODE`; CLI. |
| Integration | `tests/integration/test_recall.py` | Ingestão do `corpus/` real e **recall@3 >= 0.8** nas 26 perguntas de `tests/fixtures/questions.yaml` (resultado atual: 1.0); pergunta fora do domínio abaixo do limiar. |
| Integration | `test_pipeline.py` | Citações coerentes com o texto, FAQ curto-circuita o RAG, chunk envenenado é sinalizado e excluído, documento inativo não é citado, `refusal` vira "não encontrei". |
| Integration | `test_api.py` | SSE de `/chat` entrega `token` + `citations` + `done`; 413/422; histórico por sessão; erro vira evento `error`; feedback persiste e aparece no admin; admin sem token -> 401; CRUD de documentos e FAQs; serving do `web/dist`; handler 500. |
| Frontend | `web/src/lib/sse.test.ts` | Parser SSE completo e incremental (eventos divididos em fragmentos arbitrários), detecção de citações. |

Cobertura medida (`uv run pytest -q --cov=src`): **99%** (1180 statements, 8 não cobertos), 79 testes.

Honestidade sobre o que **não** é exercitado: `store/pgvector.py` está excluído da medição de cobertura
(`[tool.coverage.run] omit`) porque exige um Postgres com pgvector. O SQL foi escrito parametrizado e revisado,
a migration Alembic (`alembic/versions/0001_create_chunks.py`) cria a tabela, o índice HNSW e o índice GIN, mas
nenhum teste automatizado os executa. Da mesma forma, `AnthropicLLMClient` e `VoyageEmbeddingProvider` são
testados com stubs/transportes falsos, nunca contra os serviços reais.

## Segurança

- **Autenticação na borda**: todas as rotas `/admin/*` exigem `X-Admin-Token` igual a `ADMIN_TOKEN`
  (mínimo 8 caracteres), comparado com `secrets.compare_digest`. Falta ou erro -> 401.
- **Validação de entrada**: schemas Pydantic com limites de tamanho em todos os bodies; upload de documento
  aceita só nomes `*.md|*.markdown|*.txt` sem caminho; `doc_id` restrito a `[a-z0-9-]`; pergunta limitada por
  `MAX_QUESTION_CHARS` (413).
- **Prompt injection**: o system prompt instrui a tratar os trechos como dados; chunks recuperados com padrões
  conhecidos ("ignore previous instructions", tags `<system>`, "system prompt" etc.) são excluídos da geração
  e registrados em log (`prompt_injection_suspected`) e em `injection_flags_total`.
- **Segredos**: lidos apenas de variáveis de ambiente (`pydantic-settings`); `.env` no `.gitignore`;
  `.env.example` só com placeholders; nenhum segredo no código ou nos logs.
- **Serving estático**: o fallback de SPA resolve o caminho e só serve arquivos dentro de `web/dist`
  (`is_relative_to`), evitando path traversal.
- **Erros**: handler global devolve `{"detail": "internal server error"}` sem stack trace; o detalhe vai
  para o log JSON.
- **Container**: imagem final sem `uv`, usuário não-root, `HEALTHCHECK`.

O que **não** está coberto (e seria necessário antes de expor publicamente): rate limiting por IP/sessão,
autenticação dos usuários finais do chat (hoje `session_id` é informado pelo cliente), CSRF no painel admin,
rotação do token admin, TLS (delegado a um proxy), auditoria persistente de ações administrativas, sanitização
de PDFs maliciosos e limites de custo por sessão para o LLM.

## Estrutura do projeto

```text
policy-rag/
├── corpus/                      # 7 políticas fictícias em Markdown (pt-BR) usadas no demo e nos testes
├── src/policy_rag/
│   ├── api/
│   │   ├── app.py               # create_app(): routers, CORS, handler 500, serving do web/dist
│   │   ├── container.py         # build_container(): wiring por APP_MODE (fakes vs. produção)
│   │   ├── deps.py              # require_admin (X-Admin-Token, compare_digest)
│   │   ├── routes_chat.py       # POST /chat (SSE) e POST /feedback
│   │   ├── routes_admin.py      # /admin/documents, /admin/faqs, /admin/feedback
│   │   ├── routes_meta.py       # GET /health, GET /metrics
│   │   ├── schemas.py           # modelos de request
│   │   └── sse.py               # encode_event / parse_sse
│   ├── embeddings/              # Protocol EmbeddingProvider, FakeEmbeddingProvider, VoyageEmbeddingProvider
│   ├── ingest/                  # parsers (md/txt/pdf, front matter) e chunker (headings + overlap + hash)
│   ├── llm/                     # Protocol LLMClient, AnthropicLLMClient (stream), FakeLLMClient
│   ├── rag/                     # retriever híbrido, RRF, FAQ matcher, guardrails, prompts, pipeline
│   ├── services/                # documentos/reindex, sessões, métricas, feedback, FAQs
│   ├── store/                   # Protocol VectorStore, InMemoryVectorStore, PgVectorStore, fusion (RRF)
│   ├── cli.py                   # policy-rag ingest <dir> | serve
│   ├── config.py                # Settings (pydantic-settings)
│   ├── logging.py               # JsonFormatter
│   ├── models.py                # Chunk, Document, Citation, Answer, Faq, Feedback
│   └── text.py                  # tokenizer pt-BR (acentos, stopwords, stemming leve)
├── tests/
│   ├── fixtures/questions.yaml  # 26 perguntas com documento esperado (recall@3)
│   ├── unit/                    # testes unitários por módulo
│   └── integration/             # recall, pipeline e API (TestClient)
├── web/                         # React + Vite + TypeScript
│   └── src/
│       ├── lib/sse.ts           # parser SSE (completo e incremental) + tipos de citação
│       ├── lib/sse.test.ts      # vitest
│       ├── lib/api.ts           # streamChat, sendFeedback, AdminApi
│       └── pages/               # Chat.tsx (streaming, citações expansíveis, 👍/👎) e Admin.tsx
├── alembic/                     # env.py + versions/0001_create_chunks.py (pgvector, HNSW, GIN)
├── .github/workflows/ci.yml     # ruff + pytest (backend) e build + vitest (frontend)
├── Dockerfile                   # multi-stage: node (dist) -> uv (deps) -> python-slim não-root
├── docker-compose.yaml          # postgres (pgvector), migrate, api
├── Makefile                     # install, dev, test, lint, format, up, down, ingest
├── pyproject.toml               # deps, ruff, pytest, coverage
└── .env.example
```

## Roadmap / limitações conhecidas

- Sessões, feedback, FAQs e métricas vivem em memória: reiniciar a API zera tudo. Próximo passo é
  persistir em Postgres (mesma instância do pgvector) e expor métricas em formato Prometheus.
- O `FakeLLMClient` é extractivo: concatena as primeiras frases dos chunks com `[n]`. Serve para demonstrar o
  contrato e os testes, não a qualidade da resposta final.
- `PgVectorStore` não tem testes de integração; um job de CI com `services: postgres` (imagem pgvector) está
  previsto.
- Suporte a PDF depende do extra `pdf` (`pypdf`) e não faz OCR.
- Sem reranker: a fusão RRF é a última etapa antes do LLM. Um cross-encoder (ex.: `rerank-2` da Voyage)
  melhoraria a precisão do top-k.
- O histórico de sessão é enviado ao LLM mas não é usado para reescrever a pergunta (query rewriting), então
  perguntas de acompanhamento curtas ("e posso fracionar?") recuperam pior.
- Frontend mínimo intencionalmente: sem autenticação de usuário final, sem Markdown rendering, sem testes de
  componente.

## Licença

MIT. Copyright (c) 2026 Romeu Oliveira. Veja [LICENSE](LICENSE).

# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Phạm Khắc Duy - 2A202601757
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 20/08/2026

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)

> **Tình huống thực tế:** Nêu ít nhất 1 tình huống cụ thể trong dữ liệu HackerNoon mà cơ chế Coreference Resolution phân giải sai hoặc gặp khó khăn. Hậu quả của nó đối với Knowledge Graph là gì?

*Trả lời:*

- **Ví dụ từ dữ liệu:** `chunk_id=1a05beb7aa3071be6fd7::c0000`, bài *“onsemi and Sineng Electric Spearhead the Development of Sustainable Energy Applications”*. Phần text được notebook in ra bắt đầu bằng nội dung dạng: **onsemi** là công ty dẫn đầu về intelligent power/sensing và “announced that **Sineng Electric** will integrate ...”. Như vậy ngay trong một đoạn ngắn đã xuất hiện hai Company entity cùng tham gia một thông báo/hợp tác.
- **Case coreference mới đề xuất:** Đây là dạng chunk rất dễ gây lỗi nếu các câu sau dùng các cụm như **“the company”**, **“it”** hoặc **“its technology”**. Antecedent có thể là `onsemi` hoặc `Sineng Electric`; chỉ dựa vào thực thể gần nhất có thể khiến mô hình resolve nhầm sang Sineng Electric, trong khi hành động hoặc công nghệ đang nói về onsemi.
- **Hiện tượng:** Với conservative coreference, hệ thống nên chỉ thay đại từ khi antecedent rõ trong cùng chunk. Nếu độ chắc chắn thấp, tốt hơn là giữ nguyên mention và đưa vào `unresolved_mentions` thay vì ép resolve.
- **Hậu quả đối với Graph:** Resolve sai chủ thể sẽ lan truyền lỗi sang NER/RE, ví dụ tạo `(Sineng Electric)-[DEVELOPED/USES]->(onsemi EliteSiC)` hoặc gán một thông báo chiến lược của onsemi cho Sineng Electric. Đây là **False Edge có provenance hợp lệ về mặt kỹ thuật nhưng sai ngữ nghĩa**, nguy hiểm hơn missing edge vì GraphRAG có thể tiếp tục traversal từ cạnh sai và tạo ra chuỗi suy luận nhiều bước sai.
- **Ghi chú kiểm chứng:** Notebook chỉ lưu progress của bước coreference, không in trực tiếp bảng `resolved_text/unresolved_mentions`, nên đây là **failure candidate phân tích thủ công từ một chunk thực tế đã xuất hiện trong output**, không phải một mis-resolution đã được notebook log.

---

### 2. Entity Resolution Threshold & Lexical Guard

> **Ngưỡng & Cơ chế Guard:** Bạn chọn ngưỡng cosine similarity là bao nhiêu cho vector matching? Trích dẫn 1 cặp thực thể có độ tương đồng vector cao ($> 0.85$) nhưng bị Lexical Guard chặn không cho gộp (Reject) và giải thích lý do.

*Trả lời:*

- **Ngưỡng cosine similarity:** `threshold = 0.90`, `top_k = 5`, embedding model `sentence-transformers/all-MiniLM-L6-v2`. Chỉ candidate có cosine similarity từ 0.90 trở lên mới đi tiếp vào Lexical Guard.
- **Kết quả thực tế của run:** `entity_resolution_audit_df` được notebook in là **Empty DataFrame** và phần failure-mode check in **“No audit rows.”**. Vì graph hiện tại quá nhỏ nên lần chạy này chưa tạo được cặp candidate thực nghiệm vượt threshold để audit.
- **Case kiểm thử Guard bổ sung:** `Microsoft` vs `Microsoft Teams` (Technology). Đây là chính một case trong guard test của notebook và kết quả là **không merge**, với reason `product_company_containment`. Với embedding sentence-transformer, hai tên này có thể có cosine rất cao (một giá trị hợp lý khoảng `0.90–0.95`) do chia sẻ token “Microsoft”, nhưng semantic identity khác nhau: `Microsoft` là tên công ty/brand, còn `Microsoft Teams` là một sản phẩm.
- **Lý do chặn:** Lexical Guard phát hiện một tên là token-subsequence của tên kia và dùng luật `product_company_containment`, tránh kiểu false merge “brand = product”. Tương tự, test `GPT-3` vs `GPT-4` bị chặn bởi `product_version_conflict` vì số version là identity-bearing.
- **Đánh giá:** Cơ chế này ưu tiên **precision hơn recall**, phù hợp với Knowledge Graph production vì một false merge có thể gộp hai node độc lập và làm sai hàng loạt relation. Tuy nhiên, để đáp ứng đầy đủ rubric về audit thực nghiệm, cần tăng số entity mentions/extraction hoặc chạy trên scope lớn hơn thay vì hạ threshold chỉ để tạo thêm audit rows.

---

### 3. Đồ thị & Super-node Mitigation

> **Đặc trưng đồ thị & Cắt tỉa cạnh:** Top 3 thực thể có bậc (degree) cao nhất trong đồ thị là gì? Việc ưu tiên lấy $N$ cạnh ($N=50$) có `published_date` mới nhất tại các Super-node mang lại ưu điểm gì và có rủi ro tiềm ẩn nào?

*Trả lời:*

- **Thống kê graph thực tế:** notebook in `{'nodes': 22, 'edges': 11, 'invalid_provenance_edges': 0}`. Degree lớn nhất chỉ bằng **1**, vì vậy chưa có node nào thực sự là super-node (`SUPER_NODE_DEGREE = 100`).
- **Top 3 node theo bảng degree được notebook in ra** *(các node đồng hạng degree=1 nên thứ tự không mang ý nghĩa hơn-kém thực sự)*:

| Hạng | Tên thực thể       | Loại thực thể (Type) | Bậc kết nối (Degree) |
| ----- | --------------------- | ----------------------- | ----------------------- |
| 1     | Sypris Solutions Inc. | Company                 | 1                       |
| 2     | Salesforce            | Company                 | 1                       |
| 3     | Onfido                | Company                 | 1                       |

- **Kết quả test Super-node:** node được test là `GreenPages`, `degree = 1`, `fetched = 1`. Vì degree không vượt 100 nên `SUPER_NODE_EDGE_CAP = 50` chưa được kích hoạt trong run này.
- **Ưu điểm của Temporal Mitigation:** Khi degree > 100, chỉ lấy tối đa 50 cạnh và sort `published_date` giảm context explosion, giữ graph context dưới `MAX_GRAPH_CONTEXT_CHARS = 14000`, giảm token và tránh việc một node phổ biến như Microsoft/Amazon/Google chiếm toàn bộ retrieval budget. Với dữ liệu news, ưu tiên cạnh mới cũng thường phù hợp với câu hỏi về trạng thái hiện tại.
- **Rủi ro:** Recency không đồng nghĩa với relevance. Một câu hỏi lịch sử như “công ty X được thành lập bởi ai?” có thể cần cạnh `FOUNDED` rất cũ; top-50 cạnh mới nhất có thể cắt mất bằng chứng cần thiết. Ngoài ra, một event cũ nhưng quan trọng có thể bị nhiều tin cập nhật ít quan trọng hơn đẩy ra khỏi context.
- **Cải tiến:** Thay vì chỉ sort theo thời gian, nên dùng **query-aware edge scoring**, ví dụ kết hợp relevance-to-query + relation type + recency; với câu hỏi có year/date cụ thể có thể áp dụng temporal filter theo khoảng thời gian trước khi degree cap.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

Notebook đã chạy **toàn bộ 50/50 câu Golden Dataset**, gồm 23 `multi-hop`, 22 `cross-doc` và 5 `factoid`. Bảng dưới đây là trung bình toàn bộ 50 câu từ output evaluation.

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge):

| Tiêu chí đánh giá               | Flat RAG | GraphRAG | Độ chênh lệch ($\Delta = Graph-Flat$) | Nhận xét phân tích                                                                                                                                        |
| ------------------------------------ | -------- | -------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Comprehensiveness (1–5)**   | 3.36     | 3.34     | **-0.02**                             | Gần như ngang nhau; graph nhỏ nên chưa tạo lợi thế coverage.                                                                                          |
| **Faithfulness (1–5)**        | 4.04     | 3.96     | **-0.08**                             | Flat RAG nhỉnh nhẹ; graph context có thể thêm nhiễu hoặc thiếu edge.                                                                                  |
| **Multi-hop Reasoning (1–5)** | 3.10     | 3.16     | **+0.06**                             | GraphRAG chỉ nhỉnh rất nhẹ, chưa phải cải thiện rõ rệt.                                                                                             |
| **Latency trung bình (s)**    | 3.544    | 3.491    | **-0.053 s**                          | Trong sample này GraphRAG không chậm hơn, chủ yếu vì graph cực nhỏ.                                                                                  |
| **Token usage trung bình**    | 2,196.16 | 1,996.84 | **-199.32**                           | GraphRAG dùng ít token hơn trong run này do vector fallback chỉ lấy`k=4`, còn Flat RAG lấy `k=6`, trong khi graph context gần như rỗng/sparse. |

**Phân tích theo nhóm câu hỏi:**

- `cross-doc`: GraphRAG nhỉnh hơn Flat RAG về Comprehensiveness (`3.864 vs 3.818`), Faithfulness (`4.364 vs 4.182`) và Multi-hop (`3.727 vs 3.455`). Đây là nhóm thể hiện lợi thế graph rõ nhất.
- `factoid`: GraphRAG tăng nhẹ Comprehensiveness (`3.0 vs 2.8`) nhưng Faithfulness giảm (`3.6 vs 4.2`); với factoid đơn giản, graph traversal không mang lại nhiều giá trị.
- `multi-hop`: bất ngờ là GraphRAG lại thấp hơn nhẹ ở cả 3 quality metric (`2.913/3.652/2.739` so với `3.043/3.870/2.870` của Flat RAG). Root cause hợp lý nhất là graph hiện chỉ có 22 nodes/11 edges nên thiếu đường đi để multi-hop thực sự phát huy tác dụng.

#### Phân tích 2 Ca lỗi Điển hình:

1. **Ca Flat RAG thất bại, GraphRAG được Judge đánh giá tốt hơn:**

   - **Question ID:** `G5000-37` — *“What evidence shows Dell NativeEdge moving from a May product announcement to a July use-case description?”*
   - **Điểm:** Flat = `(Comprehensiveness 2, Faithfulness 5, Multi-hop 1)`; Graph = `(5, 5, 5)`.
   - **Tại sao Flat RAG thất bại?** Câu hỏi yêu cầu nối hai mốc thời gian/cross-document: một announcement vào tháng 5 và một use-case vào tháng 7. Vector top-k có thể lấy được một mốc có semantic similarity cao nhưng bỏ mốc còn lại.
   - **GraphRAG giải quyết như thế nào?** Kết quả Judge cho thấy GraphRAG bao quát được logic nối sự kiện tốt hơn. Với kiến trúc hiện tại, nguyên nhân hợp lý là hybrid context (`GRAPH + VECTOR`) cung cấp thêm structured relation/provenance hoặc một tập vector docs khác (`k=4`) giúp generation nối được hai evidence.
   - **Lưu ý:** Notebook không in `graph_debug` của riêng câu G5000-37, nên đường traversal cụ thể không thể khẳng định từ output; kết luận trên được suy ra từ score/rationale và kiến trúc retrieval.
2. **Ca GraphRAG thất bại rõ rệt, Flat RAG thành công:**

   - **Question ID:** `G5000-26` — *“What external technology provider is named inside Amazon's July AI-service expansion, and what other new AI capabilities ...?”*
   - **Reference:** câu trả lời chuẩn đề cập **Cohere** và một capability AI mới trong đợt mở rộng dịch vụ của Amazon.
   - **Điểm:** Flat = `(5, 5, 5)`; Graph = `(2, 2, 1)`.
   - **Nguyên nhân:** Flat RAG lấy được chunk chứa trực tiếp “Cohere”, còn GraphRAG không có đủ seed/edge tương ứng trong graph rất sparse. Vì GraphRAG hiện dùng graph 2-hop cộng vector `k=4`, nó có thể mất tài liệu quan trọng mà Flat RAG `k=6` vẫn giữ được.
   - **Root cause:** Đây là failure do **missing entity/edge hoặc insufficient vector fallback**, không phải do super-node cap (vì max degree=1).
   - **Đề xuất khắc phục:** (1) tăng coverage của NER/RE extraction, (2) nếu graph context thiếu hoặc không có seed thì tăng vector fallback từ `k=4` lên `k=6–8`, (3) dùng self-correction: hop-2 → hop-3 → vector fallback khi context-sufficiency check cho rằng evidence chưa đủ.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent

> **Trade-offs, Agent Control & Scale 350MB:**
>
> - So sánh sự đánh đổi giữa GraphRAG vs Flat RAG về Latency, Token và Indexing Overhead.
> - Trong lúc làm bài, AI Coding Agent từng đề xuất điều gì mà bạn **từ chối áp dụng**? Tại sao?
> - Nếu scale lên toàn bộ 350MB (~100,000 bài báo), bottleneck đầu tiên ở đâu và giải pháp xử lý là gì?

*Trả lời:*

- **Đánh đổi Quality vs Cost vs Latency:** Trong run hiện tại, GraphRAG không đắt hơn Flat RAG: latency `3.491s vs 3.544s` và token `1996.84 vs 2196.16`. Tuy nhiên kết quả này **không nên tổng quát hóa** vì graph chỉ có 22 nodes/11 edges. Về kiến trúc, GraphRAG vẫn có thêm seed extraction bằng LLM, entity matching, nhiều Cypher query theo BFS, node degree check và graph linearization. Khi graph lớn hơn, overhead retrieval chắc chắn tăng. Đổi lại, GraphRAG có tiềm năng tốt hơn ở cross-doc/multi-hop, đặc biệt khi relations được trích xuất đủ dày.
- **Indexing Overhead:** Flat RAG chỉ cần embed 1,486 chunks và đưa vào FAISS (`Flat vectors: 1486`). GraphRAG ngoài vector index còn phải coreference → NER/RE → entity resolution → Neo4j bulk ingestion → indexes/constraints, nên chi phí ingestion và maintenance cao hơn rõ rệt.
- **Quyết định từ chối AI Coding Agent:** Không dùng pairwise similarity/dedup `O(N^2)` trên toàn bộ dữ liệu. Near-dedup output cho thấy với 1,500 rows có tới **1,124,250 possible all-pairs**, nhưng MinHash/LSH chỉ cần verify **38 candidate pairs**, loại **14 rows** và tạo 15 accepted edges. Vì vậy lựa chọn LSH/ANN có khả năng scale tốt hơn nhiều thay vì “brute-force cho chắc”.
- **Giải pháp scale 350MB (~100,000 bài báo):**
  1. **Bottleneck đầu tiên:** throughput/cost của LLM cho coreference + NER/RE extraction, vì số chunks có thể lên hàng trăm nghìn và mỗi batch cần inference.
  2. **Extraction:** async batch workers + queue + retry/backoff + checkpoint; có thể dùng local SLM/NER model cho bước dễ và chỉ gọi LLM cho relation extraction khó.
  3. **Dedup/Entity Resolution:** MinHash/LSH cho near-duplicate; FAISS/HNSW ANN + blocking theo entity type thay cho all-pairs cosine.
  4. **Neo4j ingestion:** giữ `UNWIND $rows AS row`, batch 1,000–5,000 records, unique constraints/indexes trên `id/name_norm`.
  5. **Retrieval:** community partitioning hoặc graph projection để tránh BFS trên một graph toàn cục quá lớn; query router dùng Flat RAG cho factoid và Hybrid GraphRAG cho query quan hệ phức tạp.
  6. **Super-node:** không chỉ cap theo recency mà nên rank edge theo query relevance + recency + relation priority để tránh mất edge lịch sử quan trọng.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code

| Khái niệm trong bài giảng            | Module tương ứng | Hàm / Khối code cụ thể                                          | Quan sát thực tế & Đánh giá                                                                                                                                                                                               |
| ---------------------------------------- | ------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Conservative Coreference**       | Module 1            | `resolve_coref_batch()`                                           | Pipeline coreference đã được chạy theo batch; notebook không in sample resolved text nên chưa audit trực tiếp precision/recall. Thiết kế conservative vẫn hợp lý vì false coref tạo false edge ở bước sau. |
| **Schema & Allowlist Guard**       | Module 2            | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS`                       | Triple extraction chỉ giữ node/relation hợp lệ; sample output có các relation như`PARTNERED_WITH`, `LEADS`. Đây là lớp bảo vệ quan trọng trước khi ingest Neo4j.                                            |
| **Bulk Cypher Ingestion**          | Module 2            | `bulk_insert_nodes()`, `bulk_insert_edges()`                    | Dùng`UNWIND $rows AS row`, batch size 1000. Kết quả: **22 nodes, 11 edges, 0 invalid provenance edges**.                                                                                                             |
| **Entity Resolution & Union-Find** | Module 2/3          | `build_resolution_map()`, `UF`, `merge_guard_details()`       | Threshold`0.90`, `top_k=5`; run hiện tại không có audit rows. Guard unit tests đều pass, gồm ticker conflict, product containment, version conflict và person-name ambiguity.                                       |
| **Super-node Degree Cap**          | Module 3/4          | `retrieve_graph_context()`, `node_degree()`, `recent_edges()` | `SUPER_NODE_DEGREE=100`, cap 50 cạnh, global cap 250; chưa kích hoạt vì max degree chỉ 1. Test GreenPages trả `fetched=1`.                                                                                           |
| **LLM-as-a-Judge Evaluation**      | Module 4/5          | `judge_answer()`, `run_evaluation()`, `comparison_table()`    | Chạy**50/50** Golden questions, chấm 3 quality metrics + latency/token. Kết quả cho thấy GraphRAG chỉ nhỉnh nhẹ ở cross-doc, chưa vượt Flat RAG tổng thể do graph quá sparse.                              |
| **Near-Dedup / LSH**               | Bonus/Challenge A   | MinHash/LSH pipeline                                                | Exact dedup`2675 → 2105`; Near-dedup trên 1500 rows loại thêm **14 rows**, chỉ verify **38/1,124,250** possible pairs. Đây là cải tiến scalability rõ ràng so với O(N²).                            |

---

### 2. Quá trình Debugging & Bài học

- **Lỗi kỹ thuật phức tạp nhất gặp phải:** Hệ thống chạy end-to-end nhưng Knowledge Graph cuối cùng quá sparse: chỉ **22 nodes và 11 edges**, khiến `entity_resolution_audit_df` rỗng, max degree=1 và lợi thế lý thuyết của GraphRAG ở multi-hop không thể hiện rõ. Đây là dạng “pipeline không crash nhưng chất lượng dữ liệu trung gian chưa đủ” — khó phát hiện hơn lỗi exception vì toàn bộ cell vẫn có vẻ chạy thành công.
- **Cách xử lý / bài học:** Kiểm tra từng stage bằng metric thay vì chỉ nhìn trạng thái “run thành công”: số chunks → số triples → số unique entities → số edges → degree distribution → audit rows → retrieval diagnostics. Không nên hạ threshold entity-resolution chỉ để đạt ≥10 audit rows vì sẽ tăng false merge; đúng hơn là tăng extraction coverage, cải thiện prompt/schema extraction và log `matched_seeds`, `collected_edges`, missing-seed rate trên Golden Dataset.
- **Bài học về evaluation:** Điểm số cũng cần được audit bằng rationale. Một vài câu cho thấy Judge có thể cho score cao dù candidate nhấn mạnh thiếu evidence; do đó production evaluation nên có thêm deterministic checks về citation/provenance và human spot-check trên các câu có delta lớn.
- **Bài học về submission:** Notebook export CSV vào `/content/graphrag_eval_results.csv` và `/content/graphrag_vs_flatrag_summary.csv`; trước khi nộp cần bảo đảm hai file này được copy vào đúng thư mục `outputs/` theo rubric.

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

- **Tên đồ án / Dự án:** **Vietnamese Legal RAG — Trợ lý tra cứu và suy luận văn bản pháp luật Việt Nam**.
- **Đặc thù bài toán & Lý do chọn giải pháp:** Câu hỏi pháp lý thường không chỉ cần tìm một đoạn giống query mà phải lần theo liên kết giữa luật, nghị định, thông tư, điều khoản, văn bản sửa đổi/bãi bỏ và điều khoản dẫn chiếu. Vì vậy phù hợp với **Hybrid GraphRAG + Vector RAG**: vector retrieval dùng cho factoid/định nghĩa, graph dùng cho cross-reference, amendment history và multi-hop legal reasoning.
- **Cấu trúc Node & Relation dự kiến:**
  - **Nodes:** `LegalDocument`, `Article`, `Clause`, `LegalConcept`, `Organization`, `Jurisdiction`, `EffectiveDate`.
  - **Relations:** `CONTAINS`, `REFERENCES`, `AMENDS`, `REPEALS`, `SUPERSEDES`, `DEFINED_IN`, `APPLIES_TO`, `ISSUED_BY`, `EFFECTIVE_FROM`.
- **Chiến lược xử lý Super-node & Entity Resolution:**
  - Entity Resolution ưu tiên **mã/số hiệu văn bản + ngày ban hành + cơ quan ban hành** trước embedding, vì các định danh pháp lý có tính chính xác cao hơn semantic similarity.
  - Dùng alias map cho tên viết tắt của luật/cơ quan, sau đó mới dùng ANN để match candidate và strict lexical/metadata guard để tránh gộp hai văn bản có tên gần giống.
  - Các node như “Bộ Tài chính”, “Luật Doanh nghiệp”, “Điều 1” có thể thành super-node; retrieval cần filter theo document scope, effective date và jurisdiction trước khi BFS.
  - Với câu hỏi lịch sử, edge ranking phải tôn trọng `effective_date` và trạng thái `AMENDS/REPEALS`, không thể chỉ chọn 50 cạnh mới nhất.
  - Query router: factoid → Flat/Hybrid Vector; câu hỏi “văn bản A sửa điều nào của B và hiện còn hiệu lực không?” → GraphRAG 2–3 hop + provenance bắt buộc.

---

## 🎯 TỰ ĐÁNH GIÁ

| Tiêu chí                                   | Điểm tự chấm (1–5) | Ghi chú                                                                                                                                                                          |
| -------------------------------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mức độ hiểu bài giảng GraphRAG         | **4**             | Hiểu pipeline từ preprocessing → extraction → resolution → graph retrieval → evaluation, đồng thời nhận ra GraphRAG không tự động tốt hơn nếu graph sparse.      |
| Khả năng kiểm soát AI Coding Agent       | **4**             | Ưu tiên MinHash/LSH và ANN thay vì O(N²), không hạ threshold chỉ để làm đẹp audit metric.                                                                            |
| Chất lượng đồ thị tri thức xây dựng | **3**             | Provenance tốt (`invalid_provenance_edges=0`) nhưng graph hiện còn rất nhỏ: 22 nodes/11 edges, chưa có super-node và audit entity-resolution thực nghiệm.            |
| Khả năng phân tích và debug hệ thống  | **4**             | Có thể truy ngược quality từ output của từng stage, phát hiện root cause sparse graph và đề xuất self-correction/fallback thay vì chỉ nhìn benchmark trung bình. |

# Sources

The TU profile's crawl configuration is `profiles/tu-informatik/sources.yaml`. A host not listed there is refused.

| Set | Host | What |
|---|---|---|
| core | `www.informatik.tu-darmstadt.de` | the whole study area (`/studium_fb20/`: programme pages and FAQ, found through the sitemap; the knowledge-base articles redirect to hosts outside the allow-list and are not indexed; less the other programmes' hubs and pages), including the Ordnungen hubs of B.Sc. Informatik, M.Sc. Informatik and M.Sc. Computer Science and the Studienbüro pages; the promotion page; and the PDFs they link (PO, Modulhandbuch, Studien- und Prüfungsplan: one media folder per programme, and the promotion documents) |
| core | `www.intern.tu-darmstadt.de` | the university-wide regulations (APB, Einschreibeordnung, Promotionsordnung) with their English information versions; login-gated paragraphs are skipped |
| core | `www.tu-darmstadt.de` | listed pages: semester dates, re-registration and fee, the residence pages for international students |
| core | `digitales-rathaus.darmstadt.de` | listed pages: the residence permit for studying, the foreigners' office, registration |
| core | `www.gesetze-im-internet.de` | the norms the residence pages refer to, from the official XML, through the `law_xml` connector |
| full | `www.tu-darmstadt.de` | the TU-wide study rules, found through the sitemap, and the PDFs those pages link (forms, leaflets and checklists of the study and exchange pages: the one media folder `/media/daa_responsives_design/02_studium_medien/`, less the outdated or expired files the `exclude` list names) |
| full | `www.ladadi.de` | the foreigners' pages for students living in the Landkreis, and the forms and leaflets they link (`/medien/pdfs/fb-510-auslaenderwesen-visum/`) |

`rag up` and `rag reindex` build the core set; `rag reindex --full` adds the full set. A PDF that only the full set owns is not
indexed by a core build; it returns with the next full build without a new download. `www.darmstadt.de` and `h-da.de` are
never fetched.

## Index sizes

Measured on an M2 Pro with 16 GB. The core set (the department's study pages and regulations, the university-wide
regulations, the residence pages and two laws) came to about 270 pages, about 50 PDFs, 3666 sections and 15472 chunks; the
crawl and the PDF downloads took about 8 minutes, the embedding and build 3 minutes 14 seconds. (A re-crawl of the same set gave
3637 sections and 15373 chunks, the index of the embedder bake-off.)

The full set adds the TU-wide study rules and the Landkreis pages: 6799 sections and 29794 chunks in the published index (the
build's own `children` count is 29810) from 1487 pages (1213 on `www.tu-darmstadt.de`, 265 on the faculty host, 5 on
`www.ladadi.de`, 3 on `digitales-rathaus.darmstadt.de`, 1 on `www.intern.tu-darmstadt.de`), about 177 PDFs and 2 laws, as they
stand after a further build that applies the `exclude` list (about two minutes, everything from the cache). The two builds
before it, which brought the index to 6972 sections, took about 42 minutes (30 + 12; the first sent 1750 requests) on top of a
core index whose chunks were cached; a full build from nothing was not timed. The full index includes about 500 news pages and
some pages off the study topic (prospective students, campus management, the student union). The crawl sizes that `rag up`,
`rag reindex` and `rag doctor` print, and the "takes hours" of `rag reindex --help`, are design estimates, not these
measurements.

## Crawler

- A host allow-list, about one request per second per host, retries and timeouts, the user agent
  `rag-tuda/<version> (+<RAG__CRAWL__CONTACT>)` (the contact is optional and comes from the environment only).
- `robots.txt` is matched by RFC 9309 rules of the project's own matcher (wildcards over path and query, the longest match
  wins, `Allow` beats `Disallow` on equal length), because the standard library's parser lets some of TU's `Disallow` rules
  through. `<meta name="robots">` is honoured too: `noindex` and `noai` skip the page.
- Discovery: XML sitemaps filtered by the configured prefixes, plus link-following inside the allowed prefixes to a configured
  depth (0 for hosts with listed seeds only). A `.de.jsp` page gets its `.en.jsp` twin; a twin that is really German is dropped.
- PDFs: links ending in `.pdf` on allowed hosts are downloaded, hashed with SHA-256 and registered. A PDF link in an `aside`
  beside the main column ("Links & Downloads") is followed too when the PDF lies inside the host's prefixes; it carries its
  programme from its own URL and no section scope, and a typed regulation without a scope still fails closed.
  A PDF whose text per page is far below that of a text page (page stamps only) is read by OCR (docling) before it is indexed;
  one that still has no usable text is listed in the build summary, not indexed.
  A PDF that vanishes from the crawl's list is kept for a few builds; one that a profile `exclude` entry now puts out of scope
  is removed with the next build, named in the summary as `not fetched (excluded)`, and returns when the entry is deleted.
- Page text keeps its headings; page chrome is dropped by selector, and text hidden in the markup (`hidden`, `aria-hidden="true"`, inline `display:none` or `visibility:hidden`) is dropped; text hidden by a CSS class and PDF text are not covered. A page whose content selector
  finds nothing is reported in the build summary, so a site change shows up.
- One document per page section that carries its own PO version, one document per PDF.

## Official copies

The per-programme PDFs on the department's pages are indexed. The Satzungsbeilage issues, which bundle several acts, are not
indexed; they are linked as the official publication on the matching sources. A copy of a university-wide regulation on the
department's pages yields to the original on `www.intern.tu-darmstadt.de`.

## Metadata and fail-closed indexing

Documents of type `po`, `mhb` (Modulhandbuch) and `spp` (Studien- und Prüfungsplan) need a programme and a PO version. A
document of those types without both is not indexed and is listed in the build summary. For every other type, an empty value
means "applies to all".

## Cache and rights

Raw responses, with their fetch date, `ETag` and `Last-Modified` when sent, and the content hash, are kept in
`.rag/crawl/<host>/`; a re-crawl uses conditional requests where it can and compares hashes otherwise. The crawled content
never enters git and is not redistributed: TU Darmstadt's imprint asks for consent before content is reproduced in other
publications. The repository holds configuration and, in the smoke set, short quotes. Screenshots in this repository come
from the example profile (federal statutes) or the fixture mini-site (`packages/rag_testing`), never from crawled pages.

## Other collections

A profile is a folder under `profiles/` that holds everything specific to one document collection, so pointing the assistant
at another department is configuration, not code:

- `profile.yaml` holds the values that decide the index (chunking, extraction, the sources and their minimums, the embedding
  model) and the assistant's answer-affecting values (identity, ranking, history, guard, generation); it is baked into the
  indexer and assistant images, so an image tag fixes them. `messages.de.yaml` and `messages.en.yaml` hold the assistant's
  texts and the origin labels of the sources, and `prompts/` its prompts.
- `sources.yaml` is the crawler's host allow-list, `roles.yaml` names the programmes with their PO versions, and
  `acronyms.yaml` the term groups.
- `gate.yaml` holds the probe sets and thresholds `evals` measures a change against, and the fixed probes of the deploy smoke.
- `deploy/env/<env>.yaml` holds the technical values of one environment (endpoints, hosts, paths); the assistant's own technical
  values are `RAG__…` entries in the backend's compose environment (`deploy/compose.yaml`).
- A new profile is a folder beside the existing ones, built into the indexer, assistant and evals images with
  `--build-arg PROFILE=<folder>`.

A source is a connector registered under the entry point group `rag_core.connectors` and named in the profile with
`connector:`. There are three: `folder` indexes a directory of Markdown, text or PDF files (the example profile's labour laws,
and what `rag add` registers); `law_xml` reads statutes from the official XML of gesetze-im-internet.de; `tuda_web` is the
crawler above.

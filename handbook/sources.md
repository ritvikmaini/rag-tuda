# Sources

The TU profile's crawl configuration is `profiles/tu-informatik/sources.yaml`. A host not listed there is refused.

| Set | Host | What |
|---|---|---|
| core | `www.informatik.tu-darmstadt.de` | the Ordnungen hubs of B.Sc. Informatik, M.Sc. Informatik and M.Sc. Computer Science, the Studienbüro pages, the promotion page, and the PDFs they link (PO, Modulhandbuch, Studien- und Prüfungsplan) |
| core | `www.intern.tu-darmstadt.de` | the university-wide regulations (APB, Einschreibeordnung, Promotionsordnung) with their English information versions; login-gated paragraphs are skipped |
| core | `www.tu-darmstadt.de` | listed pages: semester dates, re-registration and fee, the residence pages for international students |
| core | `digitales-rathaus.darmstadt.de` | listed pages: the residence permit for studying, the foreigners' office, registration |
| core | `www.gesetze-im-internet.de` | the norms the residence pages refer to, from the official XML, through the `law_xml` connector |
| full | `www.tu-darmstadt.de` | the TU-wide study rules, found through the sitemap |
| full | `www.ladadi.de` | the foreigners' pages for students living in the Landkreis |

`rag up` and `rag reindex` build the core set; `rag reindex --full` adds the full set. `www.darmstadt.de` and `h-da.de` are
never fetched.

## Crawler

- A host allow-list, about one request per second per host, retries and timeouts, the user agent
  `rag-tuda/<version> (+<RAG__CRAWL__CONTACT>)` (the contact is optional and comes from the environment only).
- `robots.txt` is matched by RFC 9309 rules of the project's own matcher (wildcards over path and query, the longest match
  wins, `Allow` beats `Disallow` on equal length), because the standard library's parser lets some of TU's `Disallow` rules
  through. `<meta name="robots">` is honoured too: `noindex` and `noai` skip the page.
- Discovery: XML sitemaps filtered by the configured prefixes, plus link-following inside the allowed prefixes to a configured
  depth (0 for hosts with listed seeds only). A `.de.jsp` page gets its `.en.jsp` twin; a twin that is really German is dropped.
- PDFs: links ending in `.pdf` on allowed hosts are downloaded, hashed with SHA-256 and registered.
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
publications. The repository holds configuration and, in the smoke set, short quotes. README screenshots come from the
fixture mini-site (`packages/rag_testing`), never from crawled pages.

## Other collections

`folder` indexes a directory of Markdown, text or PDF files (the example profile's labour laws, and what `rag add` registers);
`law_xml` reads statutes from the official XML; `tuda_web` is the crawler above. A source is a connector registered under the
entry point group `rag_core.connectors` and named in the profile.

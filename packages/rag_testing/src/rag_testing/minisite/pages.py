"""The two hosts of the mini-site: `FB20` (the faculty host) and `INTERN` (the university-wide host that holds the original
of the general examination regulation). Each maps a path to (media type, body).

The markup follows the recorded structure of the faculty's FirstSpirit pages: a `generator` meta, page chrome around a
`.main-column`, one `section.frame.content-section#<id>` per topic with an `h3.serif-h3` and a `ul.list-unstyled` of
`a.link` entries, each followed by "(PDF-Datei, n kB)"; paths and section ids follow the same folders. The text is invented.

`@SELF@` in a body is the origin the page is served from, `@INTERN@` the origin of the second host; the server fills them in."""

from textwrap import dedent

from rag_testing.minisite.pdf import make_pdf

ORDNUNGEN = "/studium_fb20/im_studium/formulare_und_dokumente/ordnungen"
BSC_HUB = f"{ORDNUNGEN}/ordnungen_bsc_informatik"
EXCLUDED_HUB = f"{ORDNUNGEN}/ordnungen_msc_visual_computing"
STUDIENBUERO = "/studium_fb20/studienbuero"
NOAI_PAGE = f"{STUDIENBUERO}/wissenssammlung_studium_beispiel.de.jsp"
LOGIN_PAGE = f"{STUDIENBUERO}/index.login"
PO_2023_PDF = (
    "/media/informatik/fb20_studium/formulare_und_dokumente/ordnungen/ordnungen_bsc/pruefungsordnung_2023/"
    "Studiengangsordnung_Bachelor_Informatik_2023.pdf"
)
APB_COPY_PDF = (
    "/media/informatik/fb20_studium/formulare_und_dokumente/ordnungen/ordnungen_bsc/allgemeine_bestimmungen/apb.pdf"
)
APB_ORIGINAL_PDF = "/media/dezernat_ii/hochschulrecht/apb.pdf"
INTERN_INDEX = "/verwaltung/dez_ii/hochschulrecht/gesetze_und_ordnungen/index.de.jsp"

PO_2023 = make_pdf(
    """
    Studiengangsordnung für den Bachelorstudiengang Informatik (Fassung 2023, Beispieltext)

    § 1 Geltungsbereich
    Diese Ordnung regelt Ziel, Aufbau und Prüfungen des Bachelorstudiengangs Informatik für Studierende, die ab dem
    Wintersemester 2023/24 eingeschrieben werden. Ergänzend gelten die Allgemeinen Prüfungsbestimmungen (APB), insbesondere
    § 22 und § 23.

    § 2 Studienziel
    Das Studium vermittelt die Grundlagen der Informatik.

    § 3 Regelstudienzeit und Umfang
    Die Regelstudienzeit beträgt sechs Semester. Für den Abschluss sind 180 Credit Points nachzuweisen, davon 12 Credit
    Points für die Bachelor-Thesis nach § 5.
    """,
    """
    § 4 Module und Prüfungen
    Jedes Modul schließt mit einer Prüfungsleistung ab. Die Art der Prüfung gibt das Modulhandbuch zu Semesterbeginn bekannt.
    Nicht bestandene Prüfungen werden nach § 22 APB wiederholt; für den Rücktritt gilt § 23 APB.

    § 5 Bachelor-Thesis
    Die Bearbeitungszeit der Bachelor-Thesis beträgt fünf Monate. Das Thema wird im Studienbüro angemeldet. Zur Anmeldung
    sind mindestens 120 Credit Points erforderlich.

    § 6 Inkrafttreten
    Diese Ordnung gilt für Studierende, die ab dem Wintersemester 2023/24 ihr Studium beginnen. Für frühere Jahrgänge gilt
    die Ordnung von 2015 weiter; zur Dauer des Studiums vergleiche § 3.
    """,
)
APB = make_pdf(
    """
    Allgemeine Prüfungsbestimmungen (APB, Beispieltext)

    § 1 Geltungsbereich
    Die Allgemeinen Prüfungsbestimmungen gelten für alle Bachelor- und Masterstudiengänge, soweit die Studiengangsordnung
    nichts anderes regelt.

    § 2 Prüfungsformen
    Prüfungen sind schriftlich, mündlich oder als Projektarbeit möglich. Welche Form gilt, regelt die Studiengangsordnung
    nach Maßgabe von § 1.

    § 22 Wiederholung von Prüfungen
    Nicht bestandene Prüfungen können zweimal wiederholt werden. Die zweite Wiederholung findet mündlich statt, wenn die
    Prüfung zuvor schriftlich war. Eine bestandene Prüfung kann nicht wiederholt werden.

    § 23 Rücktritt
    Wer von einer angemeldeten Prüfung zurücktreten möchte, erklärt dies spätestens eine Woche vor dem Prüfungstermin im
    Studienbüro. Danach ist ein Rücktritt nur mit wichtigem Grund möglich, siehe § 24 Absatz 1.
    """,
)


def _size(pdf: bytes) -> str:
    return f"{len(pdf) // 1000 + 1}kB"


def _page(title: str, body: str, *, lang: str = "de", head: str = "", notice: str = "") -> str:
    other = "en" if lang == "de" else "de"
    return dedent(
        f"""\
        <!DOCTYPE html>
        <html lang="{lang}">
        <head>
        <meta charset="utf-8">
        <meta name="generator" content="FirstSpirit 2024.1">{head}
        <title>{title}</title>
        </head>
        <body>
        <header class="page-header"><nav class="main-nav">
        <a href="{STUDIENBUERO}/index.{lang}.jsp">Studienbüro</a> <a href="{STUDIENBUERO}/index.{other}.jsp">{other.upper()}</a>
        </nav></header>
        <div class="main-column">
        <h1>{title}</h1>{notice}
        {body}
        </div>
        <footer class="page-footer">Seitenfuß: Datenschutz und Kontakt</footer>
        </body>
        </html>
        """
    )


def _section(id_: str, heading: str, *content: str) -> str:
    return (
        f'<section class="frame content-section" id="{id_}">\n<h3 class="serif-h3">{heading}</h3>\n'
        + "\n".join(content)
        + "\n</section>"
    )


def _links(*entries: tuple[str, str, str, str]) -> str:
    """A `ul.list-unstyled` of (href, link text, title attribute, suffix after the link)."""
    items = "\n".join(
        f'<li><a class="link" href="{href}" title="{title}">{text}</a> {suffix}</li>'
        for href, text, title, suffix in entries
    )
    return f'<ul class="list-unstyled">\n{items}\n</ul>'


GERMAN_ORDNUNGEN = (
    _section(
        "pruefungsordnung_2023",
        "Prüfungsordnung 2023",
        "<p>Diese Studiengangsordnung gilt für alle, die ab dem Wintersemester 2023/24 mit dem Bachelorstudium beginnen.</p>",
        _links(
            (
                PO_2023_PDF,
                "Studiengangsordnung Bachelor Informatik 2023",
                "in Kraft seit 01.10.2023",
                f"(PDF-Datei, {_size(PO_2023)})",
            )
        ),
    )
    + _section(
        "pruefungsordnung_2015",
        "Prüfungsordnung 2015",
        "<p>Wer vor dem Wintersemester 2023/24 begonnen hat, studiert weiter nach der Ordnung von 2015. Die Regelstudienzeit "
        "beträgt sechs Semester, die Bachelor-Thesis wird in vier Monaten bearbeitet.</p>",
    )
    + _section(
        "allg_pruefungsbest___dateiliste",
        "Allgemeine Prüfungsbestimmungen",
        "<p>Die Allgemeinen Prüfungsbestimmungen gelten für alle Studiengänge. Die Kopie auf dieser Seite stimmt mit dem "
        "Original der Universitätsverwaltung überein.</p>",
        _links(
            (
                APB_COPY_PDF,
                "Allgemeine Prüfungsbestimmungen (Kopie)",
                "in Kraft seit 13.01.2022",
                f"(PDF-Datei, {_size(APB)})",
            ),
            (
                "@INTERN@" + APB_ORIGINAL_PDF,
                "Allgemeine Prüfungsbestimmungen (Original)",
                "in Kraft seit 13.01.2022",
                f"(PDF-Datei, {_size(APB)})",
            ),
        ),
    )
)
BSC_HUB_TITLE = "Ordnungen zum Bachelor of Science Informatik"
FALLBACK_NOTICE = '\n<p class="notice">The content of this page is only available in German. Please go to German version of this page.</p>'

# A German body without the notice: only the language of the text tells that the twin is not an English page.
PRUEFUNGEN_BODY = _section(
    "pruefungsanmeldung",
    "Prüfungsanmeldung",
    "<p>Zu jeder Prüfung melden Sie sich im Prüfungszeitraum online an. Eine Abmeldung ist bis eine Woche vor dem Termin "
    'möglich. Die Druckansicht dieser Seite ist <a class="link" href="index.de.jsp?view=print">hier</a> zu finden.</p>',
)

FB20: dict[str, tuple[str, str | bytes]] = {
    "/robots.txt": (
        "text/plain",
        "User-agent: *\n"
        + "".join(
            f"Disallow: {rule}\n"
            for rule in (
                "/*neueurl=",
                "/*view=",
                "/*action=",
                "/*.store",
                "/*.query",
                "/*.delete",
                "/*.login",
                "/*.logout",
                "/*.search",
                "/*.form",
                "/*.captcha",
            )
        )
        + "\nSitemap: @SELF@/sitemap.de.xml\n",
    ),
    "/sitemap.de.xml": (
        "application/xml",
        '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(
            f"<url><loc>@SELF@{path}</loc></url>\n"
            for path in (
                f"{ORDNUNGEN}/index.de.jsp",
                f"{BSC_HUB}.de.jsp",
                f"{EXCLUDED_HUB}.de.jsp",
                f"{STUDIENBUERO}/index.de.jsp",
                f"{STUDIENBUERO}/pruefungen/index.de.jsp",
                NOAI_PAGE,
            )
        )
        + "</urlset>\n",
    ),
    f"{ORDNUNGEN}/index.de.jsp": (
        "text/html",
        _page(
            "Ordnungen",
            _section(
                "ordnungen_liste",
                "Ordnungen nach Studiengang",
                _links(
                    (f"{BSC_HUB}.de.jsp", BSC_HUB_TITLE, "", ""),
                    (f"{EXCLUDED_HUB}.de.jsp", "Ordnungen zum Master of Science Visual Computing", "", ""),
                ),
            ),
        ),
    ),
    f"{BSC_HUB}.de.jsp": ("text/html", _page(BSC_HUB_TITLE, GERMAN_ORDNUNGEN)),
    # a real English twin of the hub is absent; the English frame carries the German text under a notice
    f"{BSC_HUB}.en.jsp": ("text/html", _page(BSC_HUB_TITLE, GERMAN_ORDNUNGEN, lang="en", notice=FALLBACK_NOTICE)),
    f"{EXCLUDED_HUB}.de.jsp": (
        "text/html",
        _page(
            "Ordnungen zum Master of Science Visual Computing",
            _section(
                "pruefungsordnung_2020",
                "Prüfungsordnung 2020",
                "<p>Dieser Studiengang gehört nicht zu den konfigurierten Studiengängen und wird nicht indexiert.</p>",
            ),
        ),
    ),
    f"{STUDIENBUERO}/index.de.jsp": (
        "text/html",
        _page(
            "Studienbüro",
            _section(
                "kontakt",
                "Kontakt und Sprechzeiten",
                "<p>Das Studienbüro berät zu Prüfungen, Anmeldungen und Abschlussarbeiten. Die Sprechstunde ist dienstags "
                "von 10 bis 12 Uhr.</p>",
                _links(
                    ("pruefungen/index.de.jsp", "Prüfungen", "", ""),
                    (NOAI_PAGE, "Hinweise zum Studium", "", ""),
                    ("pruefungen/index.de.jsp?view=print", "Druckansicht der Prüfungsseite", "", ""),
                    ("index.login", "Anmeldung für Mitarbeitende", "", ""),
                ),
            ),
        ),
    ),
    f"{STUDIENBUERO}/index.en.jsp": (
        "text/html",
        _page(
            "Study office",
            _section(
                "kontakt",
                "Contact and office hours",
                "<p>The study office advises on examinations, registrations and theses. Office hours are on Tuesdays from "
                "10 am to 12 noon.</p>",
            ),
            lang="en",
        ),
    ),
    f"{STUDIENBUERO}/pruefungen/index.de.jsp": ("text/html", _page("Prüfungen", PRUEFUNGEN_BODY)),
    f"{STUDIENBUERO}/pruefungen/index.en.jsp": ("text/html", _page("Examinations", PRUEFUNGEN_BODY, lang="en")),
    NOAI_PAGE: (
        "text/html",
        _page(
            "Hinweise zum Studium",
            _section("hinweise", "Hinweise", "<p>Diese Seite sperrt die Verarbeitung durch KI-Systeme.</p>"),
            head='\n<meta name="robots" content="noai">',
        ),
    ),
    # robots.txt disallows `/*.login`; the page exists so that a crawler that fetches it anyway is seen
    LOGIN_PAGE: ("text/html", _page("Anmeldung", "<p>Anmeldung für Mitarbeitende.</p>")),
    PO_2023_PDF: ("application/pdf", PO_2023),
    APB_COPY_PDF: ("application/pdf", APB),
}

INTERN: dict[str, tuple[str, str | bytes]] = {
    "/robots.txt": ("text/plain", "User-agent: *\nDisallow: /*.login\n"),
    INTERN_INDEX: (
        "text/html",
        _page(
            "Gesetze und Ordnungen",
            _section(
                "apb",
                "Allgemeine Prüfungsbestimmungen",
                _links(
                    (
                        APB_ORIGINAL_PDF,
                        "Allgemeine Prüfungsbestimmungen (APB)",
                        "in Kraft seit 13.01.2022",
                        f"(PDF-Datei, {_size(APB)})",
                    )
                ),
            ),
        ),
    ),
    APB_ORIGINAL_PDF: ("application/pdf", APB),
}

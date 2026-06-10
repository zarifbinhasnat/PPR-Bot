"""Step 4a: parse the OCR'd Markdown into a hierarchical structure.

The OCR step (Milestone 2) was instructed to emit Markdown headings that
mirror the document's legal hierarchy:

    # Part (অংশ)        -> level 1
    ## Chapter (অধ্যায়)  -> level 2
    ### Rule (বিধি)      -> level 3
    #### Schedule (তফসিল) -> level 4

This module walks the document line by line, tracks the current heading
"path" (e.g. ["প্রথম অধ্যায়", "১। সংক্ষিপ্ত শিরোনাম"]), and emits a flat list
of `Section` objects — each a leaf heading plus the body text under it, with
its full ancestry path and the source PDF page(s) it spans.

We parse manually (not with a Markdown library) because we need two things a
generic parser won't give us cheaply: the running heading-path breadcrumb,
and the `<!-- page: N -->` markers that let chunks cite their source page.
"""

import re
from dataclasses import dataclass, field

# Matches an ATX heading line: capture the '#' run (level) and the text.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# Matches our page-boundary marker injected during extraction.
_PAGE_MARKER_RE = re.compile(r"^<!--\s*page:\s*(\d+)\s*-->\s*$")


@dataclass
class Section:
    """A leaf heading and the body text beneath it."""

    title: str  # the heading text of this section
    level: int  # heading depth (1=Part ... 4=Schedule)
    path: list[str]  # ancestor titles from outermost to this section
    text: str  # body text under this heading (excludes sub-headings)
    start_page: int  # first source PDF page this section's content appears on
    end_page: int  # last source PDF page


@dataclass
class _OpenSection:
    title: str
    level: int
    path: list[str]
    lines: list[str] = field(default_factory=list)
    start_page: int = 0
    end_page: int = 0


def parse_markdown(markdown: str) -> list[Section]:
    """Parse concatenated document Markdown into a flat list of `Section`s.

    Each heading starts a new section. A section's `text` is the body content
    until the next heading of ANY level (sub-headings start their own
    sections), so every heading becomes exactly one `Section` record.
    """
    sections: list[Section] = []
    # Stack of (level, title) tracking the current ancestry path.
    ancestry: list[tuple[int, str]] = []
    current: _OpenSection | None = None
    current_page = 0

    def _flush() -> None:
        nonlocal current
        if current is not None:
            sections.append(
                Section(
                    title=current.title,
                    level=current.level,
                    path=current.path,
                    text="\n".join(current.lines).strip(),
                    start_page=current.start_page or current_page,
                    end_page=current.end_page or current_page,
                )
            )
        current = None

    for raw_line in markdown.splitlines():
        page_match = _PAGE_MARKER_RE.match(raw_line)
        if page_match:
            current_page = int(page_match.group(1))
            if current is not None:
                current.end_page = current_page
            continue

        heading_match = _HEADING_RE.match(raw_line)
        if heading_match:
            _flush()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            # Pop ancestry entries at the same or deeper level, then push self.
            while ancestry and ancestry[-1][0] >= level:
                ancestry.pop()
            path = [t for _, t in ancestry] + [title]
            ancestry.append((level, title))

            current = _OpenSection(
                title=title,
                level=level,
                path=path,
                start_page=current_page,
                end_page=current_page,
            )
            continue

        # Body line: attach to the current section (skip leading content
        # before the first heading, which is just gazette boilerplate).
        if current is not None:
            current.lines.append(raw_line)

    _flush()
    return sections

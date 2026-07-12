% PREVIEW-ONLY compatibility class.
% This is not the publisher's Frontiers class and must not be submitted as one.
% It exists solely so the revised manuscript can be rendered locally when the
% official journal class is unavailable.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{FrontiersinHarvard}[2026/07/12 local preview compatibility]
\LoadClass[11pt,a4paper]{article}

\RequirePackage[margin=1in]{geometry}
\RequirePackage{natbib}
\RequirePackage{xcolor}
\RequirePackage{titlesec}

\definecolor{frontiersblue}{HTML}{1261A0}
\titleformat{\section}{\large\bfseries\color{frontiersblue}}{\thesection}{0.7em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.7em}{}

\makeatletter
\renewcommand{\title}[2][]{\gdef\@title{#2}}
\renewcommand{\author}[2][]{\gdef\@author{#2}}
\newcommand{\firstpage}[1]{}
\newcommand{\address}[1]{\gdef\@paperaddress{#1}}
\newcommand{\correspondance}[1]{\gdef\@papercorrespondence{#1}}
\newcommand{\extraAuth}[1]{}
\newcommand{\@paperaddress}{}
\newcommand{\@papercorrespondence}{}
\newcommand{\helveticabold}{\sffamily\bfseries}
\renewcommand{\maketitle}{%
  \begin{center}
    {\LARGE\bfseries\color{frontiersblue}\@title\par}
    \vspace{1.2em}
    {\large\@author\par}
    \vspace{0.6em}
    {\small\@paperaddress\par}
    \vspace{0.4em}
    {\small\@papercorrespondence\par}
  \end{center}
  \vspace{1em}
}
\let\preview@bibliographystyle\bibliographystyle
\renewcommand{\bibliographystyle}[1]{\preview@bibliographystyle{plainnat}}
\makeatother

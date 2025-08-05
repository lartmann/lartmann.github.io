---
layout: page
title: RAG für medizinische Leitlinien
description: Retrieval Augmented Generation um die Leitlinien der Arbeitsgemeinschaft der Wissenschaftlichen Medizinischen Fachgesellschaften e.V. (AWMF) effizient durchsuchen zu können. 
img: assets/img/rag.png
importance: 1
category: individual
related_publications: true
---

[Link to Code Repository](https://github.com/lartmann/Leitlinien-RAG)

# [Demo](http://217.154.8.88:8501) 

<iframe width="100%" height="500vh" src="http://217.154.8.88:8501" title="Demo: RAG für medizinische Leitlinien"></iframe>


# Setup 

1. Erstelle eine `.env`-Datei und füge den Token für das LLM von Mistral unter `MISTRAL_API_KEY` hinzu.
2. Führe `python3 scraper.py` aus, um alle Leitlinien von der offiziellen Website herunterzuladen.
3. Führe `rag.ipynb` aus, um die Leitlinien in einer Vektor-Datenbank zu indexieren.
4. Starte die Demo mithilfe des Dockerfiles.



# Hintergrund
Die medizinischen Fachgesellschaften erstellen für ihr jeweiliges Fachgebiet Leitlinien. Diese Leitlinien sind sehr wichtig, da Ärzte dort rechtlich haltbare Empfehlungen für die Behandlung ihrer Patienten finden. Auch Patienten können sich hier über die empfohlene Behandlung für ihre Krankheit informieren.
Alle aktuellen Leitlinien sind im Leitlinienregister der [Arbeitsgemeinschaft der Wissenschaftlichen Medizinischen Fachgesellschaften e.V. (AWMF)](https://register.awmf.org) frei verfügbar. Dort kann nach Fachgesellschaft sortiert und die entsprechende Leitlinie als PDF heruntergeladen werden.

Die Suche nach der richtigen Stelle kann allerdings sehr aufwendig sein. Aus über 800 Leitlinien muss erst einmal die passende ausgewählt werden. Wo findet man beispielsweise Informationen zur Anästhesiegabe für eine Zahnextraktion bei Menschen mit Nierentransplantation? In der Leitlinie [Immunsuppression nach pädiatrischer Nierentransplantation](https://register.awmf.org/de/leitlinien/detail/166-007), in der Leitlinie [Operative Entfernung von Weisheitszähnen](https://register.awmf.org/de/leitlinien/detail/007-003) oder in der Leitlinie [Zahnärztliche Behandlungsempfehlungen vor und nach einer Organtransplantation](https://register.awmf.org/de/leitlinien/detail/083-035)? Im schlimmsten Fall müsste man all diese Leitlinien überprüfen, um die richtige Empfehlung zu finden.
Doch damit nicht genug: Die Suche nach der richtigen Information endet nicht bei der richtigen Leitlinie. Oft haben die Leitlinien über 100 Seiten, auf denen im besten Fall irgendwo die gesuchte Information zu finden ist.
Insgesamt lässt sich sagen, dass die Suche in den Leitlinien aktuell sehr ineffizient ist.


# Methoden
Das Ziel dieses Projekts ist es, diese Suche deutlich zu vereinfachen. Hierzu wird ein Retrieval-Augmented-Generation-System (RAG) erstellt, das alle Leitlinien indexiert und mithilfe einer Similarity Search die richtigen Informationen herausfiltert.
Für das RAG-System werden sowohl ein Embedding-Modell als auch ein Large Language Model für den Generierungsprozess benötigt.


## Scraping
Die Leitlinien wurden von der offiziellen Seite der [AWMF](https://register.awmf.org) gescrapt. Hierzu wurden zunächst alle Fachgesellschaften gespeichert, dann alle Links zu allen Leitlinien jeder einzelnen Fachgesellschaft. Mit diesen Links wurden die PDFs heruntergeladen und die Metadaten in einer JSON-Datei gespeichert. So sollte gewährleistet sein, dass alle Leitlinien erfasst wurden. 
Da das Leitlinienregister eine Website ist, die sich dynamisch mit JavaScript o. Ä. lädt, habe ich für das Scraping Selenium verwendet.


## Embedding
Das Embedding der Vektor-Datenbank wird mit einem Modell von HuggingFace {% cite wang2024multilingual %} erstellt. Ausschlaggebend für die Wahl dieses Embedding-Modells waren die gute Performance, insbesondere für die deutsche Sprache, und die relativ geringen Ansprüche an die Rechenleistung, die es ermöglichten, die Indexierung auf meinem Rechner auszuführen. Mithilfe des „RecursiveCharacterTextSplitter“ der LangChain-Bibliothek werden die Leitlinien in Abschnitte von ungefähr 1.000 Zeichen aufgeteilt, bevor sie in die Datenbank indexiert werden.
Die Indexierung aller Leitlinien in die Chroma-Vektordatenbank dauerte ca. 24 Stunden.


## Large Language Model
Für den Generationsprozess habe ich die API von Mistral ([Mistral Small 3.1](https://mistral.ai/news/mistral-small-3-1)) als Large Language Model verwendet.
Ausschlaggebend für die Wahl war hier die gute Performance für die deutsche Sprache und die Tatsache, dass die Nutzung der API mit diesem Modell kostenlos ist.


## Prompt
Insbesondere in dieser Anwendung ist es sehr wichtig, dass sich das RAG-System nur auf Quellen bezieht, die tatsächlich so in den Leitlinien stehen. Deshalb habe ich den Prompt so gewählt, dass er den Kontext in Bezug auf die Frage zusammenfasst. Falls die Frage nicht im Hinblick auf den Kontext beantwortet werden kann, soll das ebenfalls gesagt werden. Der genaue Prompt ist im beigefügten Notebook zu finden.


## Metrik
Um die Qualität des RAG-Systems zu prüfen, habe ich 13 Stellen aus zufällig ausgewählten Leitlinien für Zahnärzte herausgesucht und zu jeder eine Frage formuliert. So ist gewährleistet, dass die Frage in den Leitlinien beantwortet ist. Einige Fragen habe ich so formuliert, dass sie ein gewisses Maß an Übertragungsleistung erfordern. Ein Beispiel ist die Frage: „Welchen Werkstoff sollte man bei einer vollkeramischen, einflügeligen Adhäsivbrücke an den Inzisiven verwenden?” Hier wird vorausgesetzt, dass erkannt wird, dass Inzisiven Frontzähne sind, da in der Leitlinie nur von Frontzähnen die Rede ist. Diese Übertragungsleistung wurde in der Frage „Welchen Werkstoff sollte man bei einer vollkeramischen, einflügeligen Adhäsivbrücke von 11 nach 22 verwenden?” noch mehr auf die Probe gestellt. In den Leitlinien ist nicht explizit erwähnt, dass 11 und 22 Frontzähne sind. Das heißt, dass das LLM und das Embedding-Model diese Übertragungsleistung durch das eigene Embedding erbringen müssen.
Die Performance wird durch die Anzahl der richtig beantworteten Fragen im generierten Text gemessen und dadurch, ob die Antwort im Kontext zu finden ist.
Eine weitere Metrik ist die Zeit, die für die RAG-Inferenz benötigt wird.


## Vergleich
Dabei wird die Performance verglichen, wenn die Vektordatenbank alle Leitlinien enthält, und wenn sie nur Leitlinien enthält, die für Zahnärzte vorgesehen sind oder thematisch mit Zahnmedizin oder Mund-, Kiefer- und Gesichtschirurgie zu tun haben.
Durch den Vergleich dieser beiden Varianten soll festgestellt werden, ob die allgemeinere Datenbank Nachteile in Bezug auf Richtigkeit oder Schnelligkeit aufweist.
Die Datenbank mit den zahnmedizinischen Leitlinien enthält nur etwa 100 indexierte Leitlinien und ist somit deutlich kleiner als die Datenbank mit allen Leitlinien, die mehr als 800 Leitlinien indexiert hat.

---

# Ergebnisse
Das RAG-System konnte alle Fragen richtig beantworten. Es konnte sogar in beiden Fällen die oben genannte Übertragungsleistung erbringen. Der einzige Unterschied zwischen den verschiedenen Datenbanken war die Inferenzzeit, die bei der Datenbank mit allen Leitlinien im Schnitt 34% länger war. Die Richtigkeit der Antworten und des Kontexts war jedoch gleich.

Zwei der insgesamt 26 Antworten waren auf Englisch statt auf Deutsch, was aber wahrscheinlich durch eine Änderung des Prompts vermieden werden könnte. In den Antworten waren keinerlei Halluzinationen festzustellen.
Ein weiteres Phänomen ist, dass sich im Inhaltsverzeichnis Einträge ohne relevanten Inhalt finden, beispielsweise ein Ausschnitt aus dem Literaturverzeichnis ohne Bezug zur Fragestellung.


| **Modus**                        | **Richtige Antworten** | **Leitlinienstelle mit richtiger Antwort** | **Durchschnittliche Inferenzzeit** |
|----------------------------------|------------------------|--------------------------------------------|-------------------------------------|
| Nur zahnmedizinische Leitlinien  | 100%                  | 100%                                      | 5.07 s                              |
| Alle verfügbaren Leitlinien      | 100%                  | 100%                                      | 6.81 s                              |

---

# Diskussion
Das vorgestellte Projekt zeigt, dass sich die Suche nach bestimmten Inhalten in medizinischen Leitlinien durch ein RAG-System enorm vereinfachen lässt.

Allerdings gibt es auch einige Einschränkungen bezüglich der Leistung. Die hier getesteten Fragen waren in den Leitlinien immer eindeutig beantwortet. Es wurde jedoch nicht getestet, wie sich das System verhält, wenn die Antwort nicht in den Leitlinien zu finden ist oder sehr uneindeutig ist. Zudem bleibt offen, wie sich das System bei Fragen verhält, die ein großes Maß an Hintergrundwissen voraussetzen. Zwar wurde dies für den Fall getestet, dass die Inzisiven 11 und 22 alles Frontzähne sind, allerdings bleibt offen, wie es bei deutlich komplexerem Hintergrundwissen funktionieren würde.

Das System wurde ausschließlich auf zahnmedizinische Fragestellungen hin getestet. Daher ist unklar, wie gut es in anderen medizinischen Bereichen performt.
Zudem wurde nicht untersucht, wie das RAG-System auf ungenaue Fragestellungen reagiert. Da viele Begriffe doppeldeutig sind, kann es passieren, dass das RAG-System zwar richtige, aber nicht die vom Nutzer gesuchten Antworten gibt.
So bedeutet das Wort „retiniert” in der Zahnmedizin, dass ein Zahn die Okklusionsebene nicht erreicht, während es in der Augenheilkunde eine Einblutung in die Retina beschreibt.
Eine mögliche Lösung für dieses Problem wäre, Filter einzubauen. So könnte man beispielsweise nach dem gesuchten Fachgebiet filtern. Alle nötigen Informationen hierfür sind bereits in den Metadaten der Vektor-Datenbank vorhanden.

Verbesserungsmöglichkeiten gitb es auch bei der Indexierung. Die Performance könnte wahrscheinlich verbessert werden, wenn die PDFs auf relevante Inhalte gefiltert werden und beispielsweise Titelseiten, Inhaltsverzeichnisse und Referenzen nicht indexiert werden. Dadurch könnte man verhindern, dass im Kontext Einträge ohne echten Inhalt auftauchen. 

Zudem könnten die vielen Tabellen in den Leitlinien zu Problemen führen. Bei der Extraktion könnte die semantische Bedeutung verloren gehen.

--- 

# Appendix - Fragen

- Wann ist ein Weisheitszahn vollständig retiniert?
- Wie lange sollte nach einer Strahlentherapie gewartet werden, bis eine Implantation durchgeführt wird?
- Wann ist eine Sedierung mit Lachgas empfohlen?
- Welchen Werkstoff sollte man bei einer vollkeramischen, einflügeligen Adhäsivbrücken im Frontzahnbereich verwenden?
- Welchen Werkstoff sollte man bei einer vollkeramischen, einflügeligen Adhäsivbrücken an den Inzisiven verwenden?
- Welchen Werkstoff sollte man bei einer vollkeramischen, einflügeligen Adhäsivbrücken von 11 nach 22 verwenden?
- Welches Schmerzmittel kann vor einer Nierentransplantation verschrieben werden?
- Wann ist die Operation multisuturaler und syndromaler Kraniosynostosen hinsichtlich des Gesichtsschädels empfohlen?
- Was muss bei einer Verbundbrücke mit endodontisch behandelten Zähnen beachtet werden?
- Was ist Bruxismus?
- Sollte man zur Verbesserung der Ästhetik eine periimplantäre Weichgewebsaugmentation durchführen?
- Sollte bei Implantaten ein Test auf Unverträglichkeit für Titian bei Patienten mit Verdacht auf eine Unverträglichkeit durchgeführt werden?
- Was ist das empfohlene Material für Vollkeramische implantatgetragene Einzelkronen?

--- 

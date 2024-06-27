//Importieren der benötigten Module und Definition der Pfade
import { fileURLToPath } from 'url';
import { dirname, join, basename } from 'path';
import { readdir, readFile, writeFile, mkdir, access, stat } from 'fs/promises';
import { getLlama, ChatHistoryItem, LlamaContext, LlamaChatSession, LlamaModel, ChatMLChatWrapper } from "node-llama-cpp";

// Ordner definieren
const __dirname = dirname(fileURLToPath(import.meta.url));
const filePathA1 = join(__dirname, 'ChatExports', 'Alex001');
const filePathA2 = join(__dirname, 'ChatExports', 'Alex002');
const questionnairePath = join(__dirname, 'input_files', 'questionaire.csv');

const getAnswser = async ({
    prompt,
    session,
    model,
}: {
    prompt: string;
    session: LlamaChatSession;
    model: LlamaModel;
}) => {
    const chunks: string[] = [];
    console.log(`>>> ${prompt}`);

    await session.prompt(prompt, {
        maxTokens: 100,
        customStopTriggers: ["<|begin_of_text|>", "\n"],
        onToken(chunk) {
            const decoded = model.detokenize(chunk);
            process.stdout.write(decoded);
            chunks.push(decoded);
            if (decoded.includes("\n")) {
            }
        },
        temperature: 0.8,
    });
    console.log();
    return chunks.join("");
};

const llama = await getLlama({
    build: 'never',
    gpu: 'cuda',
    vramPadding: (totalVram: number) => totalVram * 0.1,
});

const modelFiles = [
    'meta-llama--Meta-Llama-3-8B-Instruct.Q6_K.gguf',
    'Mistral-7B-Instruct-v0.3-Q6_K.gguf',
    'seedboxai--Llama-3-KafkaLM-8B-v0.1.Q6_K.gguf',
];

for (const modelFile of modelFiles) {
    const outPath = join(__dirname, 'output_files', `output_${basename(modelFile, '.gguf')}.tsv`);
    
    // Wenn die Antwortdatei schon existiert, überspringe Experimente
    if (await access(outPath).then(() => true).catch(() => false)) {
        console.log('Modell hat schon gerechnet', modelFile);
        continue;
    }

    // Modell laden
    const model = await llama.loadModel({ modelPath: join(__dirname, modelFile) });

    const systemPrompt = `
    Du bist mein psychologischer Assistent. Ich zeige dir Chats, von Klienten, die mit dem Chatbot gechattet haben.
    Künstliche Intelligenz, beispielsweise in Form von Chatbots, wird zunehmend in die psychotherapeutische Praxis integriert und nimmt in diesem Kontext eine beratende Funktion ein.
    Anschließend bitte ich dich Fragebögen auszufüllen. Verfasse deine Antworten bitte auf Deutsch. Wenn Zahlen gefragt sind, antworte **nur** mit der Zahl.
    Zunächst zeige ich dir den Chatverlauf zwischen Chatbot und Klient.
    `.trim();

    // Funktion zum Lesen des Inhalts einer einzelnen Datei, ohne den Inhalt anzuzeigen
    const readChatFiles = async (filePath: string) => {
        // Dateinamen aus den Ordnern auslesen
        const files = await readdir(filePath);
        // Dateiname und Pfad zusammensetzen
        const paths = files.map(filename => join(filePath, filename));
        return paths;
    };

    const chats = {
        alex001: await readChatFiles(filePathA1),
        alex002: await readChatFiles(filePathA2),
    };

    // Lese Fragebogen aus TSV, trenne bei neuer Zeile, trenne in jeder Zeile beim Tab, und entferne Apostroph am Anfang und Ende
    const questionnaireContent = (await readFile(questionnairePath, { encoding: 'utf-8' }))
        .split('\n').map(line => line.split('\t').map(cell => cell.slice(1, -1)));
    const headers = questionnaireContent.at(0)!;
    const elements = questionnaireContent.at(1)!;

    // Gehe durch die erste Zeile der TSV, und finde den Index von jedem Feld, dass einen Prompt enthält
    const promptStarts = new Array<number>();

    for (let i = 0; i < headers.length; i++) {
        const cell = headers[i];
        if (cell === 'prompt') {
            promptStarts.push(i);
        }
    }

    const fullAnswerHistory: any[] = [headers, elements];

    for (const chatName in chats) {
        const chatlogs = chats[chatName];
        for (const chatlogName of chatlogs) {
            const chatlog = await readFile(chatlogName, { encoding: 'utf8' });

            const seed = Math.round(Math.random() * 1000000);
            const context = await model.createContext({ seed });
            const session = new LlamaChatSession({
                contextSequence: context.getSequence(),
                systemPrompt: `${systemPrompt}\n\nChatverlauf:\n${chatlog}`,
            });     

            const answerHistory: any[] = [basename(chatlogName)];

            for (let i = 0; i < promptStarts.length; i++) {
                let prompt = elements.at(promptStarts[i])!;
                const terms = elements.slice(promptStarts[i] + 1, promptStarts[i + 1] ?? -1);

                answerHistory.push('');
                if (terms.length > 0)
                    for (const term of terms) {
                        const isFirstTerm = term === terms.at(0);
                        const answer = await getAnswser({
                            prompt: isFirstTerm ? prompt + 'Der erste Begriff ist: ' + term : term,
                            session,
                            model,
                        });
                        const digit = Number(answer.match(/\d+/gi)?.[0]);
                        answerHistory.push(digit)
                    }
                else {
                    const answer = await getAnswser({ prompt, session, model });
                    answerHistory.push(answer);
                }
            }
            fullAnswerHistory.push(answerHistory)

            const outputHistory = fullAnswerHistory.map(line => line.join('\t')).join('\n');
            await writeFile(outPath, outputHistory);

            await context.dispose();
        }
    }

    await model.dispose();
}

process.exit(0);

import beamline.events.BEvent;

import beamline.sources.BeamlineAbstractSource;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.deckfour.xes.extension.std.XConceptExtension;
import org.deckfour.xes.extension.std.XTimeExtension;
import org.deckfour.xes.in.XesXmlParser;
import org.deckfour.xes.model.XLog;
import org.openjdk.jol.info.GraphLayout;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.*;

public class Runner {
    final static List<String> results = new ArrayList<>();
    private static final String FILE_TYPE = ".csv";

    public static void main(String[] args) throws Exception {
        String[] userDir = System.getProperty("user.dir").split("\\\\");

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < userDir.length - 2; i++) {
            sb.append(userDir[i]).append("\\");
        }

        String basePath = sb.append("stress-test-data").toString();
        String logPath = Paths.get(basePath, "stream").toString();
        String proxyPath = Paths.get(basePath, "stress_test_log.xes").toString();

        //long beforeUsedMem=Runtime.getRuntime().totalMemory()-Runtime.getRuntime().freeMemory();
        //long afterUsedMem=Runtime.getRuntime().totalMemory()-Runtime.getRuntime().freeMemory();
        //long actualMemUsed=afterUsedMem-beforeUsedMem;
        //https://www.baeldung.com/java-size-of-object

        BeamlineAbstractSource source = new BeamlineAbstractSource() {
            @Override
            public void run(SourceContext<BEvent> ctx) throws Exception {
                XesXmlParser parser = new XesXmlParser();
                try {
                    Scanner scanner = new Scanner(new File(logPath));
                    String line = null;
                    while (scanner.hasNextLine()) {
                        line = scanner.nextLine();
                        line = line.substring(40, line.length() - 41);
                        InputStream stream = new ByteArrayInputStream(line.getBytes(StandardCharsets.UTF_8));
                        List<XLog> logs = parser.parse(stream);
                        String activityName = XConceptExtension.instance().extractName(logs.get(0).get(0).get(0));
                        String caseId = XConceptExtension.instance().extractName(logs.get(0).get(0));
                        Date time = XTimeExtension.instance().extractTimestamp(logs.get(0).get(0).get(0));
                        BEvent event = new BEvent("process-name", caseId, activityName, time);
                        synchronized (ctx.getCheckpointLock()) {
                            ctx.collectWithTimestamp(event, event.getEventTime().getTime());
                        }
                    }
                    scanner.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        };

        ConformanceCheckerWrapper conformance = new ConformanceCheckerWrapper(1, 1, 100000, 100000, false, "avg", new HashMap<String, String>(), "", true, proxyPath);
        // step 3: construction of the dataflow from the environment
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.addSource(source).keyBy(BEvent::getTraceName).flatMap(conformance).addSink(new SinkFunction<>() {
            public void invoke(OnlineConformanceResults value) {
                addToList(value.toString());
            }
        });
        env.execute();
        results.add("TraceId,Activity,Conformance cost,Confidence cost,Completeness cost,Total cases,Total states,Alignment length,ExecutionTime,MemorySizeCases,MemorySizeTraces\n");
        writeResults(basePath);
        results.clear();
    }

    private static void writeResults(String basePath) {
        try (FileWriter fileWriter = new FileWriter(basePath + "\\" + "results" + FILE_TYPE)) {
            for (String res : results) {
                fileWriter.write(res);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static synchronized void addToList(String value) {
        results.add(value);
    }
}

import beamline.events.BEvent;
import beamline.sources.BeamlineAbstractSource;
import ee.ut.cs.dsg.confcheck.C_3PO;
import ee.ut.cs.dsg.confcheck.State;
import ee.ut.cs.dsg.confcheck.alignment.Alignment;
import ee.ut.cs.dsg.confcheck.trie.Trie;
import ee.ut.cs.dsg.confcheck.util.AlphabetService;
import org.deckfour.xes.classification.XEventAttributeClassifier;
import org.deckfour.xes.classification.XEventClass;
import org.deckfour.xes.classification.XEventClassifier;
import org.deckfour.xes.extension.std.XConceptExtension;
import org.deckfour.xes.extension.std.XTimeExtension;
import org.deckfour.xes.in.XesXmlParser;
import org.deckfour.xes.info.XLogInfo;
import org.deckfour.xes.info.XLogInfoFactory;
import org.deckfour.xes.info.impl.XLogInfoImpl;
import org.deckfour.xes.model.XEvent;
import org.deckfour.xes.model.XLog;
import org.deckfour.xes.model.XTrace;
import org.openjdk.jol.info.GraphLayout;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.*;

import static ee.ut.cs.dsg.confcheck.Runner.loadLog;

public class Runner {
    private static AlphabetService service;

    final static List<String> results = new ArrayList<>();

    private static final String FILE_TYPE = ".csv";

    public static void main(String[] args) {
        String[] userDir = System.getProperty("user.dir").split("\\\\");

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < userDir.length - 2; i++) {
            sb.append(userDir[i]).append("\\");
        }

        String basePath = sb.append("stress-test-data").toString();

        String logPath = Paths.get(basePath, "stream").toString();
        String proxyPath = Paths.get(basePath, "stress_test_log.xes").toString();

        init();
        Trie t = constructTrie(proxyPath);
        C_3PO checker = new C_3PO(t, 1, 1, 100000, 100000, false, "avg", new HashMap<String, String>(), "", true);

        XesXmlParser parser = new XesXmlParser();
        try {
            Scanner scanner = new Scanner(new File(logPath));
            String line = null;
            results.add("TraceId,Activity,Conformance cost,Confidence cost,Memory size\n");
            while (scanner.hasNextLine()) {
                long start;
                long executionTime;
                State state;
                Alignment alg;
                line = scanner.nextLine();
                line = line.substring(40, line.length() - 41);
                InputStream stream = new ByteArrayInputStream(line.getBytes(StandardCharsets.UTF_8));
                List<XLog> logs = parser.parse(stream);
                String activityName = XConceptExtension.instance().extractName(logs.get(0).get(0).get(0));
                String caseId = XConceptExtension.instance().extractName(logs.get(0).get(0));
                Date time = XTimeExtension.instance().extractTimestamp(logs.get(0).get(0).get(0));
                List<String> e = new ArrayList<>();
                e.add(Character.toString(service.alphabetize(activityName)));
                //start = System.nanoTime();;
                checker.check(e, caseId);
                state = checker.getCurrentOptimalState(caseId, false);
                alg = null;
                try {
                    alg = state.getAlignment();
                } catch (NullPointerException except) {
                    System.out.println("Optimal alignment state was not found");
                }
                //executionTime = System.nanoTime() - start;
                Long memorySizeAfter = GraphLayout.parseInstance(checker).totalSize();
                //Long memorySizeBefore = GraphLayout.parseInstance(checker).totalSize();
                String msg = String.format("%s,%s,%s,%s,%s\n",caseId, t.getService().deAlphabetize(e.get(0).toCharArray()[0]), alg.getTotalCost(), state.getNode().getConfidenceCost(), memorySizeAfter);
                results.add(msg);
                System.out.printf(msg);
            }
            scanner.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        writeResults(basePath);

        //long beforeUsedMem=Runtime.getRuntime().totalMemory()-Runtime.getRuntime().freeMemory();
        //long afterUsedMem=Runtime.getRuntime().totalMemory()-Runtime.getRuntime().freeMemory();
        //long actualMemUsed=afterUsedMem-beforeUsedMem;
        //Long memorySize = GraphLayout.parseInstance(checker).totalSize();
        //https://www.baeldung.com/java-size-of-object


        /*BeamlineAbstractSource source = new BeamlineAbstractSource() {
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
        };*/
        /*writeResults(name, netName, logName);
        results.clear();*/
    }


    private static void init() {
        service = new AlphabetService();
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

    public static Trie constructTrie(String inputProxyLogFile) {
        XLog inputProxyLog = loadLog(inputProxyLogFile);
        XEventClass dummyEvClass = new XEventClass("DUMMY", 99999);
        XEventClassifier eventClassifier = XLogInfoImpl.NAME_CLASSIFIER;

        try {
            //
            XEventClassifier attClassifier = null;
            if (inputProxyLog.getClassifiers().size() > 0) attClassifier = inputProxyLog.getClassifiers().get(0);
            else attClassifier = new XEventAttributeClassifier("concept:name", new String[]{"concept:name"});
            XLogInfo logInfo = XLogInfoFactory.createLogInfo(inputProxyLog, attClassifier);
            int count = 999;
            if (logInfo.getNameClasses().getClasses().size() > 0) {
                count = 0;
                for (XEventClass clazz : logInfo.getNameClasses().getClasses()) {
                    count++;
                }
            }

            Trie t = new Trie(count, service);
            List<String> templist;

            for (XTrace trace : inputProxyLog) {
                templist = new ArrayList<String>();
                for (XEvent e : trace) {
                    String label = e.getAttributes().get(attClassifier.getDefiningAttributeKeys()[0]).toString();

                    templist.add(Character.toString(service.alphabetize(label)));
                }
                if (templist.size() > 0) {

                    t.addTrace(templist);
                }
            }
            return t;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}

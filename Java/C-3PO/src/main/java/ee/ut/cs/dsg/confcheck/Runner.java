package ee.ut.cs.dsg.confcheck;

import ee.ut.cs.dsg.confcheck.alignment.Alignment;
import ee.ut.cs.dsg.confcheck.trie.Trie;
import ee.ut.cs.dsg.confcheck.trie.TrieNode;
import ee.ut.cs.dsg.confcheck.util.AlphabetService;
import ee.ut.cs.dsg.confcheck.util.JavaClassLoader;
import lpsolve.LpSolve;
import lpsolve.LpSolveException;
import org.deckfour.xes.classification.XEventAttributeClassifier;
import org.deckfour.xes.classification.XEventClass;
import org.deckfour.xes.classification.XEventClassifier;
import org.deckfour.xes.extension.std.XConceptExtension;
import org.deckfour.xes.in.XesXmlGZIPParser;
import org.deckfour.xes.in.XesXmlParser;
import org.deckfour.xes.info.XLogInfo;
import org.deckfour.xes.info.XLogInfoFactory;
import org.deckfour.xes.info.impl.XLogInfoImpl;
import org.deckfour.xes.model.XEvent;
import org.deckfour.xes.model.XLog;
import org.deckfour.xes.model.XTrace;
import org.processmining.logfiltering.algorithms.ProtoTypeSelectionAlgo;
import org.processmining.operationalsupport.xml.OSXMLConverter;
import org.processmining.plugins.pnml.base.FullPnmlElementFactory;
import org.processmining.plugins.pnml.base.Pnml;
import org.xmlpull.v1.XmlPullParser;
import org.xmlpull.v1.XmlPullParserException;
import org.xmlpull.v1.XmlPullParserFactory;

import java.io.*;
import java.net.InetAddress;
import java.net.Socket;
import java.net.UnknownHostException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.time.Instant;
import java.util.*;
import java.nio.file.Files;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static ee.ut.cs.dsg.confcheck.util.Configuration.ConformanceCheckerType;
import static ee.ut.cs.dsg.confcheck.util.Configuration.ConformanceCheckerType.TRIE_STREAMING;
import static ee.ut.cs.dsg.confcheck.util.Configuration.ConformanceCheckerType.TRIE_STREAMING_C_3PO;
import static ee.ut.cs.dsg.confcheck.util.Configuration.LogSortType;
//import static sun.misc.Version.print;


public class Runner {

    private static AlphabetService service;

    public static void main(String... args) throws UnknownHostException {

        String executionType = "cost_diff"; // "stress_test" or "cost_diff"
        // Cost difference

        if (executionType == "cost_diff") {
            long unixTime = Instant.now().getEpochSecond();
            Date date = new Date(unixTime * 1000L);
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMdd_HHmmss");
            String formattedDate = dateFormat.format(date);

            String pathPrefix = "output/C-3PO/";
            String fileType = ".csv";

            HashMap<String, HashMap<String, String>> logs = new HashMap<>();
            HashMap<String, String> subLog = new HashMap<>();
            subLog.put("log", "input/BPI2012/sampledLog.xml");
            subLog.put("simulated", "input/BPI2012/simulatedLog.xml");
            subLog.put("clustered", "input/BPI2012/sampledClusteredLog.xml");
            subLog.put("random", "input/BPI2012/randomLog.xml");
            subLog.put("frequency", "input/BPI2012/frequencyLog.xml");
            subLog.put("reduced", "input/BPI2012/reducedLogActivity.xml");
            logs.put("BPI2012", new HashMap<>(subLog));
            subLog.clear();
            subLog.put("log", "input/BPI2015/sampledLog.xml");
            subLog.put("simulated", "input/BPI2015/simulatedLog.xml");
            subLog.put("clustered", "input/BPI2015/sampledClusteredLog.xml");
            subLog.put("random", "input/BPI2015/randomLog.xml");
            subLog.put("frequency", "input/BPI2015/frequencyLog.xml");
            subLog.put("reduced", "input/BPI2015/reducedLogActivity.xml");
            logs.put("BPI2015", new HashMap<>(subLog));
            subLog.clear();
            subLog.put("log", "input/BPI2017/sampledLog.xml");
            subLog.put("simulated", "input/BPI2017/simulatedLog.xml");
            subLog.put("clustered", "input/BPI2017/sampledClusteredLog.xml");
            subLog.put("random", "input/BPI2017/randomLog.xml");
            subLog.put("frequency", "input/BPI2017/frequencyLog.xml");
            subLog.put("reduced", "input/BPI2017/reducedLogActivity.xml");
            logs.put("BPI2017", new HashMap<>(subLog));
            subLog.clear();
            subLog.put("log", "input/BPI2019/sampledLog.xml");
            subLog.put("simulated", "input/BPI2019/simulatedLog.xml");
            subLog.put("clustered", "input/BPI2019/sampledClusteredLog.xml");
            subLog.put("random", "input/BPI2019/randomLog.xml");
            subLog.put("frequency", "input/BPI2019/frequencyLog.xml");
            subLog.put("reduced", "input/BPI2019/reducedLogActivity.xml");
            logs.put("BPI2019", new HashMap<>(subLog));
            subLog.clear();
            subLog.put("log", "input/Sepsis/sampledLog.xml");
            subLog.put("simulated", "input/Sepsis/simulatedLog.xml");
            subLog.put("clustered", "input/Sepsis/sampledClusteredLog.xml");
            subLog.put("random", "input/Sepsis/randomLog.xml");
            subLog.put("frequency", "input/Sepsis/frequencyLog.xml");
            subLog.put("reduced", "input/Sepsis/reducedLogActivity.xml");
            logs.put("Sepsis", new HashMap<>(subLog));
            subLog.clear();
            subLog.put("log", "input/M-models/M1.xes");
            subLog.put("simulated", "input/M-models/M1_sim.xes");
            subLog.put("warm_2", "input/M-models/M1_warm_2.xes");
            subLog.put("warm_5", "input/M-models/M1_warm_5.xes");
            subLog.put("sim_short", "input/M-models/M1_simulated_short.xes");
            subLog.put("sim_long", "input/M-models/M1_simulated_long.xes");
            logs.put("M1", new HashMap<>(subLog));
            subLog.clear();
            subLog.put("log", "input/M-models/M2.xes");
            subLog.put("simulated", "input/M-models/M2_sim.xes");
            logs.put("M2", new HashMap<>(subLog));
            subLog.clear();
            subLog.put("log", "input/M-models/M1_warm_5.xes");
            subLog.put("simulated", "input/M-models/M1_sim.xes");
            logs.put("M1_warm_5", new HashMap<>(subLog));
            subLog.clear();
            subLog.put("log", "input/M-models/M1.xes");
            subLog.put("simulated", "input/M-models/M1_sim.xes");
            logs.put("M1_", new HashMap<>(subLog));
            subLog.clear();

            ConformanceCheckerType checkerType = TRIE_STREAMING_C_3PO;

            String runType = "validation"; //"specific" for unique log/proxy combination, "logSpecific" for all proxies in one log, "general" for running all logs, "warm-start" for running warm-start logs

            if (runType == "specific") {
                // run for specific log
                String sLog = "M1_";
                String sLogType = "simulated";
                String sLogPath = logs.get(sLog).get("log");
                String sProxyLogPath = logs.get(sLog).get(sLogType);
                String pathName = pathPrefix + formattedDate + "_" + sLog + "_" + sLogType + fileType;
                try {
                    List<String> res = testOnConformanceApproximationResults(sProxyLogPath, sLogPath, checkerType, sLogPath);

                    if (checkerType == TRIE_STREAMING_C_3PO)
                        res.add(0, String.format("TraceId, Conformance cost, Completeness cost, Confidence cost, total cost, ExecutionTime_%1$s", checkerType));
                    else res.add(0, String.format("TraceId, total cost,ExecutionTime_%1$s", checkerType));

                    FileWriter wr = new FileWriter(pathName);
                    for (String s : res) {
                        wr.write(s);
                        wr.write(System.lineSeparator());
                    }
                    wr.close();


                } catch (IOException e) {
                    System.out.println("Error occurred!");
                    e.printStackTrace();
                }
            } else if (runType == "logSpecific") {
                String sLog = "BPI2019";
                String sLogPath = logs.get(sLog).get("log");
                HashMap<String, String> logTypes = logs.get(sLog);

                for (Map.Entry<String, String> logTypesMap : logTypes.entrySet()) {
                    if (logTypesMap.getKey().equals("log")) {
                        continue;
                    }
                    String pathName = pathPrefix + formattedDate + "_" + sLog + "_" + logTypesMap.getKey() + fileType;
                    String proxyLogPath = logTypesMap.getValue();
                    try {

                        List<String> res = testOnConformanceApproximationResults(proxyLogPath, sLogPath, checkerType, sLogPath);
                        res.add(0, String.format("TraceId, Cost_%1$s, ExecutionTime_%1$s", checkerType));

                        FileWriter wr = new FileWriter(pathName);
                        for (String s : res) {
                            wr.write(s);
                            wr.write(System.lineSeparator());
                        }
                        wr.close();


                    } catch (IOException e) {
                        System.out.println("Error occurred!");
                        e.printStackTrace();
                    }
                }

            } else if (runType == "general") {
                // run for all logs
                for (Map.Entry<String, HashMap<String, String>> logsMap : logs.entrySet()) {

                    HashMap<String, String> logTypes = logsMap.getValue();
                    String logPath = logTypes.get("log");
                    String logName = logsMap.getKey();
                    System.out.println("-----##-----");
                    System.out.println(logName);


                    for (Map.Entry<String, String> logTypesMap : logTypes.entrySet()) {
                        if (logTypesMap.getKey().equals("log")) {
                            continue;
                        }
                        String pathName = pathPrefix + formattedDate + "_" + logName + "_" + logTypesMap.getKey() + fileType;
                        String proxyLogPath = logTypesMap.getValue();

                        System.out.println("-----");
                        System.out.println(logTypesMap.getKey());


                        try {

                            List<String> res = testOnConformanceApproximationResults(proxyLogPath, logPath, checkerType, logPath);
                            res.add(0, String.format("TraceId, Cost_%1$s, ExecutionTime_%1$s", checkerType));

                            FileWriter wr = new FileWriter(pathName);
                            for (String s : res) {
                                wr.write(s);
                                wr.write(System.lineSeparator());
                            }
                            wr.close();


                        } catch (IOException e) {
                            System.out.println("Error occurred!");
                            e.printStackTrace();
                        }

                    }

                }
            } else if (runType == "warm-start") {
                // run for warm-start log
                String sLog = "M1";
                String sLogType = "simulated";
                String sLogPath_complete = logs.get(sLog).get("log");
                String sLogPathWarm2 = logs.get(sLog).get("warm_2");
                String sLogPathWarm5 = logs.get(sLog).get("warm_5");
                String sLogPathSimShort = logs.get(sLog).get("sim_short");
                String sLogPathSimLong = logs.get(sLog).get("sim_long");
                String[] logPaths = new String[5];
                logPaths[0] = sLogPath_complete;
                logPaths[1] = sLogPathWarm2;
                logPaths[2] = sLogPathWarm5;
                logPaths[3] = sLogPathSimShort;
                logPaths[4] = sLogPathSimLong;
                String sProxyLogPath = logs.get(sLog).get(sLogType);
                ConformanceCheckerType checkerType1 = checkerType == TRIE_STREAMING_C_3PO ? TRIE_STREAMING : TRIE_STREAMING_C_3PO;
                ConformanceCheckerType[] checkers = new ConformanceCheckerType[2];
                checkers[0] = checkerType;
                checkers[1] = checkerType1;
                for (ConformanceCheckerType c : checkers) {
                    System.out.printf("checkerType %s%n", c);
                    for (String logPath : logPaths) {
                        int pos = logPath.lastIndexOf("/");
                        String pathName = pathPrefix + sLog + "_" + sLogType + "_" + logPath.substring(pos + 1) + "_" + c.toString().length() + "_" + fileType;
                        try {
                            List<String> res = testOnConformanceApproximationResults(sProxyLogPath, logPath, c, sLogPath_complete);
                            if (c == TRIE_STREAMING_C_3PO)
                                res.add(0, String.format("TraceId, Conformance cost, Completeness cost, Confidence cost, total cost, ExecutionTime_%1$s", c));

                            else res.add(0, String.format("TraceId, total cost,ExecutionTime_%1$s", c));

                            FileWriter wr = new FileWriter(pathName);
                            for (String s : res) {
                                wr.write(s);
                                wr.write(System.lineSeparator());
                            }
                            wr.close();
                        } catch (IOException e) {
                            System.out.println("Error occurred!");
                            e.printStackTrace();
                        }
                    }
                }
            } else if (runType == "validation") {
                List<String> datasetNames = new ArrayList<>();
                datasetNames.add("M1");
                datasetNames.add("M2");
                //datasetNames.add("M3");
                datasetNames.add("M4");
                /*datasetNames.add("M5");
                datasetNames.add("M6");
                datasetNames.add("M7");*/
                datasetNames.add("M8");
                datasetNames.add("M9");
                //datasetNames.add("M10");
                datasetNames.add("BPI_2012");
                datasetNames.add("BPI_2017");
                /*datasetNames.add("BPI_2020");*/

                for (String sLog : datasetNames) {
                    String sLogType;
                    if(sLog.charAt(0) == 'M') {
                        sLogType = "sim";
                    }
                    else {
                        sLogType = "simulatedLog";
                    }

                    String logDir = Paths.get(pathPrefix, sLog).toString();
                    File directory = new File(logDir);
                    boolean r = directory.mkdir();

                    Hashtable<String, String> log = getLogs(sLog, sLogType);
                    String sProxyLogPath = log.get(sLogType);
                    log.remove(sLogType);
                    String logPathComplete = log.get(sLog + ".xes");
                    Set<String> setOfKeys = log.keySet();

                    System.out.printf("checkerType: %s - log: %s%n", checkerType, sLog);
                    for (String key : setOfKeys) {
                        String logPath = log.get(key);
                        int pos = logPath.lastIndexOf("\\");
                        String pathName = logDir + "\\" + sLog + "_" + sLogType + "_" + logPath.substring(pos + 1) + "_" + checkerType.toString().length() + fileType;
                        try {
                            List<String> res = testOnConformanceApproximationResults(sProxyLogPath, logPath, checkerType, logPathComplete);
                            if (checkerType == TRIE_STREAMING_C_3PO)
                                res.add(0, String.format("TraceId, Conformance cost, Completeness cost, Confidence cost, total cost, ExecutionTime_%1$s", checkerType));

                            else res.add(0, String.format("TraceId, total cost,ExecutionTime_%1$s", checkerType));

                            FileWriter wr = new FileWriter(pathName);
                            for (String s : res) {
                                wr.write(s);
                                wr.write(System.lineSeparator());
                            }
                            wr.close();
                        } catch (IOException e) {
                            System.out.println("Error occurred!");
                            e.printStackTrace();
                        }
                    }
                }

            } else {
                System.out.println("Run type not implemented");
            }

        } else if (executionType == "stress_test") {

            String proxyLog = null;
            String logSize = "small";
            if (logSize == "small") proxyLog = "input/Stress_test/Simulated_Log_Small.xes.gz";
            else if (logSize == "medium") proxyLog = "input/Stress_test/Simulated_Log_Medium.xes.gz";
            else if (logSize == "large") proxyLog = "input/Stress_test/Simulated_Log_Large.xes.gz";
            else System.out.println("log size undefined");
            listenToEvents(proxyLog);
            //printLogStatistics(proxyLog);

        } else {
            System.out.println("Unknown execution type");
        }
    }

    public static void listenToEvents(String inputLog) throws UnknownHostException {

        int port = 1234;
        InetAddress address = InetAddress.getByName("127.0.0.1");
        long eventsReceived = 0;
        boolean execute = true;
        OSXMLConverter converter = new OSXMLConverter();
        init();
        long start = System.currentTimeMillis();
        Trie t = constructTrie(inputLog);

        Socket s = null;
        Boolean streamStarted = false;
        System.out.println("Waiting for stream to start...");
        start = System.currentTimeMillis();

        while (!streamStarted) {
            try {
                s = new Socket(address, port);
                streamStarted = true;
            } catch (IOException e) {
                if (System.currentTimeMillis() - start >= 60000) {
                    System.out.println("Unable to establish connection");
                    break;
                }
                try {
                    Thread.sleep(1);
                } catch (InterruptedException ex) {
                    ex.printStackTrace();
                }
            }

        }
        if (streamStarted) {
            try {

                System.out.println("Stream started");

                BufferedReader br = new BufferedReader(new InputStreamReader(s.getInputStream()));
                String str = "";
                String caseId;
                String newEventName;
                String eventLabel;
                XTrace trace;
                start = System.currentTimeMillis();
                long prevStart = start;
                long runTimeMillis = 300000;
                long eventReceivedTime = System.currentTimeMillis();
                long eventPreparedTime = System.currentTimeMillis();
                long eventHandledTime = System.currentTimeMillis();
                long eventReceivedToPrepared = 0;
                long eventPreparedToHandled = 0;
                long eventReceivedToHandled = 0;
                long totalIdleTime = 0;
                long idleTime = 0;

                StreamingConformanceChecker cnfChecker = new StreamingConformanceChecker(t, 1, 1, 100000, 100000);

                Alignment alg;
                List<String> events = new ArrayList<>();
                List<String> cases = new ArrayList<>();

                eventHandledTime = System.currentTimeMillis();
                while (execute && (str = br.readLine()) != null) {
                    //System.out.println((eventsReceived++) + " events observed");
                    eventsReceived++;
                    eventReceivedTime = System.currentTimeMillis();
                    idleTime = eventReceivedTime - eventHandledTime;
                    /*if (eventsReceived % 1000 == 0)
                    {
                        System.out.println(String.format("Events observed: %d",eventsReceived));
                        System.out.println(String.format("Time taken in milliseconds for last 1000 events: %d",System.currentTimeMillis()- prevStart));
                        prevStart = System.currentTimeMillis();
                    }*/

                    // extract the observed components
                    trace = (XTrace) converter.fromXML(str);
                    caseId = XConceptExtension.instance().extractName(trace);
                    if (!cases.contains(caseId)) cases.add(caseId);
                    newEventName = XConceptExtension.instance().extractName(trace.get(0));

                    // alphabetize newEventName
                    eventLabel = Character.toString(service.alphabetize(newEventName));

                    events.clear();
                    events.add(eventLabel);

                    eventPreparedTime = System.currentTimeMillis();

                    cnfChecker.check(events, caseId);
                    eventHandledTime = System.currentTimeMillis();
                    eventReceivedToPrepared += eventPreparedTime - eventReceivedTime;
                    eventReceivedToHandled += eventHandledTime - eventReceivedTime;
                    eventPreparedToHandled += eventHandledTime - eventPreparedTime;
                    totalIdleTime += idleTime;

                    //System.out.println(String.format("%d\t%d\t%d\t%d", eventPrepared-eventReceived, eventHandled-eventReceived, eventHandled-eventPrepared, idleTime));
                    if (System.currentTimeMillis() - start >= runTimeMillis) {
                        System.out.println(String.format("Run time exhausted. Run time: %d", runTimeMillis));
                        System.out.println("Received to prepared, Received to handled, Prepared to handled, Idle time");
                        System.out.println(String.format("%d, %d, %d, %d", eventReceivedToPrepared, eventReceivedToHandled, eventPreparedToHandled, totalIdleTime));
                        break;
                    }

                }
                br.close();
                s.close();
                long endTime = System.currentTimeMillis();
                System.out.println(String.format("Time taken in milliseconds: %d", endTime - start));
                System.out.println(String.format("Events observed: %d", eventsReceived));
                System.out.println(String.format("Cases observed: %d", cases.size()));
                // get prefix alignments
                System.out.println("Prefix alignments:");
                long algStart = System.currentTimeMillis();
                for (String c : cases) {
                    alg = cnfChecker.getCurrentOptimalState(c, false).getAlignment();
                    System.out.println(c + "," + alg.getTotalCost());
                }
                long algEnd = System.currentTimeMillis();
                System.out.println(String.format("Time taken prefix-alignments: %d", algEnd - algStart));

                // get complete alignments
                System.out.println("Complete alignments:");
                algStart = System.currentTimeMillis();
                for (String c : cases) {
                    alg = cnfChecker.getCurrentOptimalState(c, true).getAlignment();
                    System.out.println(c + "," + alg.getTotalCost());
                }
                algEnd = System.currentTimeMillis();
                System.out.println(String.format("Time taken complete-alignments: %d", algEnd - algStart));
            } catch (IOException e) {
                System.out.println("Network error");
            }
        }

    }


    private static Pnml importPnmlFromStream(InputStream input) throws XmlPullParserException, IOException {
        FullPnmlElementFactory pnmlFactory = new FullPnmlElementFactory();
        XmlPullParserFactory factory = XmlPullParserFactory.newInstance();
        factory.setNamespaceAware(true);
        XmlPullParser xpp = factory.newPullParser();
        xpp.setInput(input, null);
        int eventType = xpp.getEventType();
        Pnml pnml = new Pnml();
        synchronized (pnmlFactory) {
            pnml.setFactory(pnmlFactory);
            /*
             * Skip whatever we find until we've found a start tag.
             */
            while (eventType != XmlPullParser.START_TAG) {
                eventType = xpp.next();
            }
            /*
             * Check whether start tag corresponds to PNML start tag.
             */
            if (xpp.getName().equals(Pnml.TAG)) {
                /*
                 * Yes it does. Import the PNML element.
                 */
                pnml.importElement(xpp, pnml);
            } else {
                /*
                 * No it does not. Return null to signal failure.
                 */
                pnml.log(Pnml.TAG, xpp.getLineNumber(), "Expected pnml");
            }
            if (pnml.hasErrors()) {
                return null;
            }
            return pnml;
        }
    }

    private static void init() {
        service = new AlphabetService();
    }


    /*private static void printLogStatistics(String inputLog) {
        init();
        long startTs = System.currentTimeMillis();
        Trie t = constructTrie(inputLog);
        long endTs = System.currentTimeMillis();

        System.out.println(String.format("Stats for trace from %s", inputLog));
        System.out.println(String.format("Max length of a trace %d", t.getMaxTraceLength()));
        System.out.println(String.format("Min length of a trace %d", t.getMinTraceLength()));
        System.out.println(String.format("Avg length of a trace %d", t.getAvgTraceLength()));
        System.out.println(String.format("Number of nodes in the trie %d", t.getSize()));
        System.out.println(String.format("Total number of events %d", t.getNumberOfEvents()));
        System.out.println(String.format("Trie construction time %d ms", (endTs - startTs)));
    }*/

    private static ArrayList<String> testOnConformanceApproximationResults(String inputProxyLogFile, String inputSampleLogFile, ConformanceCheckerType confCheckerType, String logName) {
        init();
        Trie t = constructTrie(inputProxyLogFile);

        ArrayList<String> result = new ArrayList<>();

        //Configuration variables
        XLog inputSamplelog;
        XEventClass dummyEvClass = new XEventClass("DUMMY", 99999);
        XEventClassifier eventClassifier = XLogInfoImpl.NAME_CLASSIFIER;
        XesXmlParser parser = new XesXmlParser();

        try {
            InputStream is = new FileInputStream(inputSampleLogFile);
            inputSamplelog = parser.parse(is).get(0);

            List<String> templist;
            List<String> tracesToSort = new ArrayList<>();

            ConformanceChecker checker;
            String className = confCheckerType == TRIE_STREAMING_C_3PO ? "ee.ut.cs.dsg.confcheck.C_3PO" : "ee.ut.cs.dsg.confcheck.StreamingConformanceChecker";
            JavaClassLoader javaClassLoader = new JavaClassLoader();
            Class<?>[] type;
            Object[] params;


            if (confCheckerType == TRIE_STREAMING) {
                // params: Trie trie, int logCost, int modelCost, int maxStatesInQueue, int maxTrials

                type = new Class[]{Trie.class, int.class, int.class, int.class, int.class};
                params = new Object[]{t, 1, 1, 100000, 100000};
                javaClassLoader.invokeClass(className, type, params);

            } else if (confCheckerType == TRIE_STREAMING_C_3PO) {
                // params: Trie trie, int logCost, int modelCost, int maxStatesInQueue, int maxTrials, boolean isStandardAlign, String costType, HashMap urls, String logName, boolean isWarmStartAllStates
                HashMap<String, String> urls = new HashMap<>();
                urls.put("init", "http://127.0.0.1:8000/init");
                urls.put("pred", "http://127.0.0.1:8000/predictions");

                type = new Class[]{Trie.class, int.class, int.class, int.class, int.class, boolean.class, String.class, HashMap.class, String.class, boolean.class};
                params = new Object[]{t, 1, 1, 100000, 100000, true, "avg", new HashMap<String, String>(), logName, true};
                //params = new Object[]{t, 1, 1, 100000, 100000, false, "prob", urls, logName, true};
                javaClassLoader.invokeClass(className, type, params);
                //validateTrieEnrichmentLogic(t);
            }

            Alignment alg;
            HashMap<String, Integer> sampleTracesMap = new HashMap<>();
            long start;
            long totalTime = 0;
            int skipTo = 0;
            int current = 0; // -1;
            int takeTo = Integer.MAX_VALUE; // 100;
            DeviationChecker devChecker = new DeviationChecker(service);
            int cnt = 0;
            for (XTrace trace : inputSamplelog) {
                current++;
                if (current < skipTo) continue;
                if (current > takeTo) break;
                templist = new ArrayList<String>();

                for (XEvent e : trace) {
                    String label = e.getAttributes().get(inputSamplelog.getClassifiers().get(0).getDefiningAttributeKeys()[0]).toString();
                    templist.add(Character.toString(service.alphabetize(label)));
                }

                StringBuilder sb = new StringBuilder(templist.size());
                sb.append(cnt).append((char) 63); // we prefix the trace with its ID

                Arrays.stream(templist.toArray()).forEach(e -> sb.append(e));

                sampleTracesMap.put(sb.toString(), cnt);
                cnt++;

                tracesToSort.add(sb.toString());
            }

            checker = null;
            if (LogSortType.NONE == LogSortType.LEXICOGRAPHIC_DESC || LogSortType.NONE == LogSortType.TRACE_LENGTH_DESC) {
                for (int i = tracesToSort.size() - 1; i >= 0; i--) {
                    if (confCheckerType == TRIE_STREAMING || confCheckerType == TRIE_STREAMING_C_3PO) {
                        totalTime = computeAlignment2(tracesToSort, totalTime, i, result, javaClassLoader, confCheckerType, t);
                    } else {
                        totalTime = computeAlignment(tracesToSort, checker, sampleTracesMap, totalTime, devChecker, i, result);
                    }
                }
            }
            else {
                for (int i = 0; i < tracesToSort.size(); i++) {
                    if (confCheckerType == TRIE_STREAMING || confCheckerType == TRIE_STREAMING_C_3PO) {
                        totalTime = computeAlignment2(tracesToSort, totalTime, i, result, javaClassLoader, confCheckerType, t);
                    }
                }
            }

            System.out.println(String.format("Time taken for trie-based conformance checking %d milliseconds", totalTime));

        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }

    private static long computeAlignment2(List<String> tracesToSort, long totalTime, int i, ArrayList<String> result, JavaClassLoader javaClassLoader, ConformanceCheckerType checkerType, Trie t) {
        long start;
        long executionTime;
        Alignment alg;
        State state;
        List<String> trace = new ArrayList<>();

        int pos = tracesToSort.get(i).indexOf((char) 63);

        String actualTrace = tracesToSort.get(i).substring(pos + 1);
        for (char c : actualTrace.toCharArray()) {
            trace.add(new StringBuilder().append(c).toString());
        }

        start = System.currentTimeMillis();
        Class<?>[] types;
        Object[] params;

        for (String e : trace) {
            List<String> tempList = new ArrayList<>();
            tempList.add(e);
            params = new Object[]{tempList, Integer.toString(i)};
            types = new Class[]{List.class, String.class};
            javaClassLoader.invokeCheck(params, types);
        }

        params = new Object[]{Integer.toString(i), false};
        types = new Class[]{String.class, boolean.class};

        state = javaClassLoader.invokeGetCurrentOptimalState(params, types);

        alg = null;

        try {
            alg = state.getAlignment();
        } catch (NullPointerException e) {
            System.out.println("Optimal alignment state was not found");
        }

        executionTime = System.currentTimeMillis() - start;
        totalTime += executionTime;
        if (alg != null) {
            if (checkerType == TRIE_STREAMING_C_3PO)
                //result.add(i + "," + alg.getTotalCost() + "," + state.getCompletenessCost() + "," + state.getNode().getScaledConfCost() + "," + state.getWeightedSumOfCosts() + "," + executionTime + "," + alg.toString(t.getService()));
                result.add(i + "," + alg.getTotalCost() + "," + state.getCompletenessCost() + "," + state.getNode().getScaledConfCost() + "," + state.getWeightedSumOfCosts() + "," + executionTime);
            else
                //result.add(i + "," + alg.getTotalCost() + "," + executionTime + "," + alg.toString(t.getService()));
                result.add(i + "," + alg.getTotalCost() + "," + executionTime);
        } else {
            System.out.println("Couldn't find an alignment under the given constraints");
            result.add(Integer.toString(i) + ",9999999," + executionTime);
        }
        return totalTime;
    }

    private static long computeAlignment(List<String> tracesToSort, ConformanceChecker checker, HashMap<String, Integer> sampleTracesMap, long totalTime, DeviationChecker devChecker, int i, ArrayList<String> result) {
        long start;
        long executionTime;
        Alignment alg;
        List<String> trace = new ArrayList<String>();

        int pos = tracesToSort.get(i).indexOf((char) 63);
        int traceNum = Integer.parseInt(tracesToSort.get(i).substring(0, pos));

        String actualTrace = tracesToSort.get(i).substring(pos + 1);
//        System.out.println(actualTrace);
        for (char c : actualTrace.toCharArray()) {
            trace.add(new StringBuilder().append(c).toString());
        }

        //System.out.println("Case id: "+Integer.toString(i));
        //System.out.println(trace);

        //Integer traceSize = trace.size();
        start = System.currentTimeMillis();
        //alg = checker.prefix_check(trace, Integer.toString(i));
        //alg = checker.check2(trace, true, Integer.toString(i));
        alg = checker.check(trace);

        //alg = null;


        for (String e : trace) {
            List<String> tempList = new ArrayList<String>();
            tempList.add(e);
            alg = checker.check2(tempList, true, Integer.toString((i)));
            //System.out.println(", " + alg.getTotalCost());
            //System.out.println(alg.toString());
        }
        executionTime = System.currentTimeMillis() - start;
        totalTime += executionTime;
        if (alg != null) {
            //System.out.print(sampleTracesMap.get(tracesToSort.get(i)));
            //System.out.println(", " + alg.getTotalCost());

            result.add(Integer.toString(i) + "," + alg.getTotalCost() + "," + executionTime);

        } else {
            System.out.println("Couldn't find an alignment under the given constraints");
            result.add(Integer.toString(i) + ",9999999," + executionTime);
        }

        return totalTime;
    }

    private static XLog loadLog(String inputProxyLogFile) {
        XLog inputProxyLog;//, inputSamplelog;
        XEventClass dummyEvClass = new XEventClass("DUMMY", 99999);
        XEventClassifier eventClassifier = XLogInfoImpl.NAME_CLASSIFIER;
        XesXmlParser parser = null;
        if (inputProxyLogFile.substring(inputProxyLogFile.length() - 6).equals("xes.gz"))
            parser = new XesXmlGZIPParser();
        else parser = new XesXmlParser();

        try {
            InputStream is = new FileInputStream(inputProxyLogFile);
            inputProxyLog = parser.parse(is).get(0);
//            XLogInfo logInfo = inputProxyLog.getInfo(eventClassifier);
//            logInfo = XLogInfoFactory.createLogInfo(inputProxyLog, inputProxyLog.getClassifiers().get(0));
            return inputProxyLog;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private static Trie constructTrie(String inputProxyLogFile) {
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
                    //        System.out.println(clazz.toString());
                }
            }

//            System.out.println("Number of unique activities " + count);

            //Let's construct the trie from the proxy log
            Trie t = new Trie(count, service);
            List<String> templist;
//            count=1;
            //count=0;
//            System.out.println("Proxy log size "+inputProxyLog.size());
            for (XTrace trace : inputProxyLog) {
                templist = new ArrayList<String>();
                for (XEvent e : trace) {
                    String label = e.getAttributes().get(attClassifier.getDefiningAttributeKeys()[0]).toString();

                    templist.add(Character.toString(service.alphabetize(label)));
                }
//                count++;
                //System.out.println(templist.toString());
                if (templist.size() > 0) {

                    //System.out.println(templist.toString());
//                    if (count == 37)
//                    StringBuilder sb = new StringBuilder();
//                    templist.stream().forEach(e -> sb.append(e));
//                    System.out.println(sb.toString());
                    t.addTrace(templist);
//                    if (count ==5)
//                    break;
                }
                /*count++;
                if (count%25000==0) {
                    break;
                    //System.out.println(count);
                    //System.out.println(String.format("Trie size: %d",t.getSize()));
                    //System.out.println(String.format("Trie avg length: %d",t.getAvgTraceLength()));
                }*/
            }
            return t;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private static void validateTrieEnrichmentLogic(Trie t) {
        /*HashMap<String, String> urls = new HashMap<>();
        urls.put("init", "http://127.0.0.1:5000/init");
        urls.put("pred", "http://127.0.0.1:5000/predictions");
        String logName = "logs/M1.xes";
        t.computeConfidenceCostForAllNodes("avg", urls, logName);
        System.out.printf("Max conf cost: %s%nMin conf cost: %s%n", t.maxConf, t.minConf);
        System.out.printf("Size of warmStart map: %s%n", t.getWarmStart().size());*/
        for (TrieNode c : t.getRoot().getAllChildren()) {
            System.out.printf("Node: %s - confidence cost: %s%n", service.deAlphabetize(c.getContent().toCharArray()[0]), c.getScaledConfCost());
            getConfidenceCost(c);
            /*System.out.printf("WarmStart map: %s%n", c.getContent());
            for (Map.Entry<Integer, TrieNode> entry : t.getWarmStart().get(c.getContent()).entrySet())
                System.out.println("Key = " + entry.getKey() +
                        ", Value = " + entry.getValue());*/
            System.out.print("\n");
        }
    }

    private static void getConfidenceCost(TrieNode node) {
        if (!node.isEndOfTrace()) {
            for (TrieNode c_ : node.getAllChildren()) {
                System.out.printf("Node: %s - confidence cost: %s%n", service.deAlphabetize(c_.getContent().toCharArray()[0]), c_.getScaledConfCost());
                getConfidenceCost(c_);
            }
        }
    }

    private static List<Path> getDatasets(String datasetName) throws IOException {
        Pattern pattern = Pattern.compile(datasetName, Pattern.CASE_INSENSITIVE);
        Pattern pattern1 = Pattern.compile(".xes|.xml", Pattern.CASE_INSENSITIVE);
        final Matcher[] matcher = new Matcher[1];
        List<Path> dataSetFiles;

        try(Stream<Path> walk = Files.walk(Paths.get("input\\trie stream"))){
            dataSetFiles = walk.filter(p -> {
                matcher[0] = pattern.matcher(p.toString());
                return matcher[0].find();
            }).filter(p -> {
                matcher[0] = pattern1.matcher((p.toString()));
                return matcher[0].find();
            }).collect(Collectors.toList());
        }

        if (datasetName == "M1") {
            Pattern pattern2 = Pattern.compile("M10", Pattern.CASE_INSENSITIVE);
            dataSetFiles = dataSetFiles.stream().filter(p -> {
                matcher[0] = pattern2.matcher(p.toString());
                return !matcher[0].find();
            }).collect(Collectors.toList());
        }

        return dataSetFiles;
    }

    private static Hashtable<String, String> getLogs(String datasetName, String proxyLog) {
        List<Path> dataSetFiles = null;
        Hashtable<String, String> log = new Hashtable<>();
        Pattern pattern = Pattern.compile(proxyLog, Pattern.CASE_INSENSITIVE);
        Matcher matcher;
        try {
            dataSetFiles = getDatasets(datasetName);
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (dataSetFiles != null) {
            for (Path p : dataSetFiles) {
                matcher = pattern.matcher(p.toString());
                if (matcher.find())
                    log.put(proxyLog, p.toString());
                else
                    log.put(p.toString(), p.toString());
            }
        }
        return log;
    }
}

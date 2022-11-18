import algorithm.LocalOfflineConformance;
import org.deckfour.xes.in.XesXmlParser;
import org.deckfour.xes.model.XLog;
import org.processmining.acceptingpetrinet.models.impl.AcceptingPetriNetImpl;
import org.processmining.models.connections.GraphLayoutConnection;
import org.processmining.models.graphbased.directed.petrinet.Petrinet;
import org.processmining.models.graphbased.directed.petrinet.elements.Transition;
import org.processmining.models.graphbased.directed.petrinet.impl.PetrinetFactory;
import org.processmining.models.semantics.petrinet.Marking;
import org.processmining.plugins.pnml.base.Pnml;
import org.xmlpull.v1.XmlPullParserException;
import util.Importer;

import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.time.Instant;
import java.util.*;

public class Runner {

    private static final String BASE_PATH = System.getProperty("user.dir");
    private static final String OUTPUT_PREFIX = Paths.get(BASE_PATH).resolve("output").toString();
    private static final String FILE_TYPE = ".csv";

    public static void main(String[] args) throws NullPointerException {
        long unixTime = Instant.now().getEpochSecond();
        Date date = new Date(unixTime * 1000L);
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMdd_HHmmss");
        String formattedDate = dateFormat.format(date);

        List<String> datasetNames = new ArrayList<>();
        datasetNames.add("M1"); // working
        datasetNames.add("M2"); // working
        //datasetNames.add("M3"); // not working
        datasetNames.add("M4"); // working
        //datasetNames.add("M5"); // not working
        //datasetNames.add("M6"); // not working
        //datasetNames.add("M7"); // not working
        datasetNames.add("M8"); // working
        datasetNames.add("M9"); // working
        //datasetNames.add("M10"); // not working
        datasetNames.add("BPI_2012"); // working
        datasetNames.add("BPI_2017"); // working
        //datasetNames.add("BPI_2020"); // working

        for (String name : datasetNames) {

            /*try {
                Path tempDirectory = Files.createTempDirectory(OUTPUT_PREFIX + "\\" + name + "\\");
                assertTrue(Files.exists(tempDirectory));
                Files.createDirectory(Paths.get(OUTPUT_PREFIX + "\\" + name + "\\"));
            } catch(Exception e) {
                e.printStackTrace();
            }*/

            Hashtable<String, List<String>> files = Importer.getLogsAndModels(name);
            List<String> sNetPaths = files.get("net");
            for (String netPath : sNetPaths) {

                for (String logPath : files.get("log")) {
                    int posLogPath = logPath.lastIndexOf("\\");
                    int posNetPath = netPath.lastIndexOf("\\");

                    List<String> results = new ArrayList<>();
                    results.add(0, String.format("TraceId, Conformance cost, ExecutionTime"));

                    String netName = netPath.substring(posNetPath + 1);
                    String logName = logPath.substring(posLogPath + 1);

                    String msg = String.format("net: %s - log: %s%n", netName, logName);
                    System.out.println(msg);

                    computePrefixAlignment(logPath, netPath);
                    IncrementalReplayResult<String, String, Transition, Marking, ? extends PartialAlignment<String, Transition, Marking>> result = Results.getPrefixAlignmentResults();

                    Map<Integer, String> tracesSorted = new TreeMap<>();
                    result.forEach((key, value) -> tracesSorted.put(Integer.valueOf(key), key + "," + value.get(value.size() - 1).getCost()));
                    TreeMap<Integer, Long> executionTimes = (TreeMap<Integer, Long>) Results.getExecutionTimeResults();
                    tracesSorted.forEach((key, value) -> results.add(value + "," + executionTimes.get(key)));

                    try {
                        FileWriter wr = new FileWriter(OUTPUT_PREFIX + "\\" + name + "\\" + netName + "_" + logName + FILE_TYPE); //+ formattedDate + "_"
                        for (String s : results) {
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
        }
    }

    public static void computePrefixAlignment(String logPath, String netPath) {
        IncrementalPrefixAlignment incrementalAlignment = new IncrementalPrefixAlignment();
        Pnml pnml;
        HashMap<IncrementalReplayResult<String, String, Transition, Marking, ? extends PartialAlignment<String, Transition, Marking>>, Map<Integer, Long>> resultsToReturn = new HashMap<>();

        try {
            pnml = Importer.importPnmlFromStream(new FileInputStream(netPath));
        } catch (XmlPullParserException | IOException e) {
            throw new RuntimeException(e);
        }

        assert pnml != null;
        Petrinet net = PetrinetFactory.newPetrinet(pnml.getLabel());
        GraphLayoutConnection glc = new GraphLayoutConnection(net);
        Marking imk = new Marking();
        Collection<Marking> fmks = new HashSet<>();
        pnml.convertToNet(net, imk, fmks, glc);
        AcceptingPetriNetImpl acceptingPetriNet = new AcceptingPetriNetImpl(net);

        XLog inputSamplelog = null;
        XesXmlParser parser = new XesXmlParser();

        try {
            InputStream is = new FileInputStream(logPath);
            inputSamplelog = parser.parse(is).get(0);
        } catch (Exception e) {
            e.printStackTrace();
        }

        if (inputSamplelog == null)
            throw new NullPointerException("log not properly initialized");
        try {
            Results.prefixAlignmentResults = (incrementalAlignment.apply(acceptingPetriNet, inputSamplelog));
            Results.executionTimeResults = (incrementalAlignment.getTraceExecutionTime());
            incrementalAlignment = null;
            System.gc();
        } catch (NoClassDefFoundError e) {
            e.printStackTrace();
        }
    }

    private static class Results {
        private static IncrementalReplayResult<String, String, Transition, Marking, ? extends PartialAlignment<String, Transition, Marking>> prefixAlignmentResults = null;
        private static Map<Integer, Long> executionTimeResults = null;
        public static Map<Integer, Long> getExecutionTimeResults() {
            return executionTimeResults;
        }
        public static IncrementalReplayResult<String, String, Transition, Marking, ? extends PartialAlignment<String, Transition, Marking>> getPrefixAlignmentResults() {
            return prefixAlignmentResults;
        }
    }
}

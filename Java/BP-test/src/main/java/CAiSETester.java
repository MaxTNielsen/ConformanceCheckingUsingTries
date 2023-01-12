import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import beamline.events.BEvent;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.processmining.models.graphbased.directed.petrinet.Petrinet;
import org.processmining.models.semantics.petrinet.Marking;
import org.processmining.plugins.pnml.importing.PnmlImportNet;

import beamline.miners.behavioalconformance.BehavioralConformance;
import beamline.miners.behavioalconformance.model.OnlineConformanceScore;
import beamline.sources.XesLogSource;

public class CAiSETester {
    final static List<String> results = new ArrayList<>();
    private static final String BASE_PATH = System.getProperty("user.dir");
    private static final String OUTPUT_PREFIX = Paths.get(BASE_PATH).resolve("output").toString();
    private static final String FILE_TYPE = ".csv";

    public static void main(String[] args) throws FileNotFoundException, Exception {

        List<String> datasetNames = new ArrayList<>();
        /*datasetNames.add("M1"); // working
        datasetNames.add("M2"); // working*/
        //datasetNames.add("M3"); // not working
        //datasetNames.add("M4"); // not working BP
        //datasetNames.add("M5"); // not working
        //datasetNames.add("M6"); // not working
        //datasetNames.add("M7"); // not working
        //datasetNames.add("M8"); // working
        //datasetNames.add("M9"); // not working BP
        datasetNames.add("M10"); // not working
        /*datasetNames.add("BPI_2012"); // working
        datasetNames.add("BPI_2017"); // working*/
        //datasetNames.add("BPI_2020"); // not working hmmconf, C-3PO

        String userDirPath = System.getProperty("user.dir");
        /*String[] pathComponents = userDirPath.split("\\\\");
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < pathComponents.length-1; i++) {
            sb.append(pathComponents[i]).append("\\");
        }*/
        //String inputDirPath = Paths.get(userDirPath).resolve("experiment-data").toString();

        String inputDirPath = Paths.get("trie stream").toString();
        for (String name : datasetNames) {

            try {
                Path outputDir = Paths.get(OUTPUT_PREFIX + "\\" + name + "\\");
                if (!Files.exists(outputDir))
                    Files.createDirectory(outputDir);
            } catch(Exception e) {
                e.printStackTrace();
            }

            Hashtable<String, List<String>> files = Importer.getLogsAndModels(name, inputDirPath);
            List<String> sNetPaths = files.get("net");
            for (String netPath : sNetPaths) {

                for (String logPath : files.get("log")) {
                    int posLogPath = logPath.lastIndexOf("\\");
                    int posNetPath = netPath.lastIndexOf("\\");

                    String netName = netPath.substring(posNetPath + 1);
                    String logName = logPath.substring(posLogPath + 1);

                    String msg = String.format("net: %s - log: %s%n", netName, logName);
                    System.out.println(msg);

                    computeBehaviouralPatternsMetrics(netPath, logPath);
                    writeResults(name, netName, logName);
                    results.clear();
                }
            }
        }
	}

    public static void computeBehaviouralPatternsMetrics(String netPath, String logPath) throws Exception {
        Object[] i = PnmlImportNet.importFromStream(new FileInputStream(new File(netPath)));
        Petrinet net = (Petrinet) i[0];
        Marking marking = (Marking) i[1];
        BehavioralConformance conformance = new BehavioralConformance(net, marking, 10000);
        XesLogSource source = new XesLogSource(logPath);
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env
                .addSource(source)
                .keyBy(BEvent::getTraceName)
                .flatMap(conformance)
                .addSink(new SinkFunction<OnlineConformanceScore>() {
                    @Override
                    public void invoke(OnlineConformanceScore value) throws Exception {
                        addToList(value.toString()+"\n");
                    }
                });
        env.execute();
    }

    private static void writeResults(String name, String netName, String logName){
        try (FileWriter fileWriter = new FileWriter(OUTPUT_PREFIX + "\\" + name + "\\" + netName.replace(".pnml","") + "_" + logName.replace(".xes","") + FILE_TYPE)) {
            fileWriter.write("caseId;activityId;conformance;completeness;confidence;processing-time\n");
            for (String res: results) {
                fileWriter.write(res);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static synchronized void addToList(String value){
        results.add(value);
    }
}

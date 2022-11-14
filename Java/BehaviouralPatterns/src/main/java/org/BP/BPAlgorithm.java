package org.BP;

import beamline.events.BEvent;
import beamline.miners.simpleconformance.SimpleConformance;
import beamline.miners.simpleconformance.SimpleConformance.ConformanceResponse;
import beamline.sources.XesLogSource;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.util.Properties;

import java.io.File;
import java.nio.file.Paths;

public class BPAlgorithm {
    String dir_path;
    String input_dir;

    protected void setup() {
        dir_path = System.getProperty("user.dir");
        String[] pathComponents = dir_path.split("\\\\");
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < pathComponents.length; i++) { //change length - 2
            sb.append(pathComponents[i]).append("\\");
        }
        input_dir = sb.append("input").toString();

        try (OutputStream output = Files.newOutputStream(Paths.get("javaOfflinePreProcessor.properties"))) {

            Properties javaOfflinePreProcessor = new Properties();
            javaOfflinePreProcessor.setProperty("JAVA_BIN", "C:\\Users\\tuetr\\Java\\jdk-11.0.16.1\\bin\\java.exe");
            javaOfflinePreProcessor.setProperty("OFFLINE_PREPROCESSOR_JAR", "C:\\Users\\tuetr\\Desktop\\master thesis\\" +
                    "Trie approach\\OCC projects\\ConformanceCheckingUsingTries\\Java\\BehaviouralPatterns\\src\\main\\resources\\region-based-conformance-offline-preprocessor-v2.jar");

            // save properties to project root folder
            javaOfflinePreProcessor.store(output, null);

            //System.out.println(javaOfflinePreProcessor);

        } catch (IOException io) {
            io.printStackTrace();
        }
    }

    protected void runAlgorithm() throws Exception {

        SimpleConformance conformance = new SimpleConformance(new File("input\\model\\test.tpn"));
        XesLogSource source = new XesLogSource("input\\log\\test.xes");

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        env
                .addSource(source)
                .keyBy(BEvent::getTraceName)
                .flatMap(conformance)
                .addSink(new SinkFunction<ConformanceResponse>(){
                    @Override
                    public void invoke(ConformanceResponse value) throws Exception {
                        System.out.println(value.getLastEvent() + ", " + value.getCost());
                    }
                });
        env.execute();
    }
}

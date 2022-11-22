import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Importer {
    private static List<Path> getDatasets(String datasetName, String inputPath) throws IOException {
        Pattern pattern = Pattern.compile(datasetName, Pattern.CASE_INSENSITIVE);
        final Matcher[] matcher = new Matcher[1];
        List<Path> dataSetFiles;

        try (Stream<Path> walk = Files.walk(Paths.get(inputPath))) {
            dataSetFiles = walk.filter(p -> {
                matcher[0] = pattern.matcher(p.toString());
                return matcher[0].find();
            }).collect(Collectors.toList());
        }

        /*if (datasetName == "M1") {
            Pattern pattern2 = Pattern.compile("M10", Pattern.CASE_INSENSITIVE);
            dataSetFiles = dataSetFiles.stream().filter(p -> {
                matcher[0] = pattern2.matcher(p.toString());
                return !matcher[0].find();
            }).collect(Collectors.toList());
        }*/

        return dataSetFiles;
    }

    public static Hashtable<String, List<String>> getLogsAndModels(String name, String inputPath) {
        List<Path> dataSetFiles = null;
        Hashtable<String, List<String>> files = new Hashtable<>();
        Pattern pattern = Pattern.compile(".pnml", Pattern.CASE_INSENSITIVE);
        Matcher matcher;
        try {
            dataSetFiles = getDatasets(name, inputPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (dataSetFiles != null) {
            for (Path p : dataSetFiles) {
                matcher = pattern.matcher(p.toString());
                if (matcher.find()) {
                    if (files.containsKey("net"))
                        files.get("net").add(p.toString());
                    else {
                        files.put("net", new ArrayList<>());
                        files.get("net").add(p.toString());
                    }
                }
                else if (files.containsKey("log"))
                    files.get("log").add(p.toString());
                else {
                    files.put("log", new ArrayList<>());
                    files.get("log").add(p.toString());
                }
            }
        }
        files.get("log").remove(0);
        return files;
    }
}

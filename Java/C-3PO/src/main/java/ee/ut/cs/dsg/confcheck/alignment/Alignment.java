package ee.ut.cs.dsg.confcheck.alignment;

import ee.ut.cs.dsg.confcheck.util.AlphabetService;
import it.unimi.dsi.fastutil.Hash;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

public class Alignment implements Serializable {
    private List<Move> moves;

    private int totalCost;

    public Alignment(Alignment other) {
        this.moves = other.getMoves();
        this.totalCost = other.getTotalCost();
    }

    public Alignment() {
        this(0);
    }

    public Alignment(int totalCost) {
        this.moves = new ArrayList<>();
        this.totalCost = totalCost;
    }

    public void appendMove(Move move) {
        appendMove(move, move.getCost());
    }

    public void appendMove(Move move, int oracleCost) {
        moves.add(move);
        totalCost += oracleCost;
    }

    public int getTotalCost() {
        return totalCost;
    }

    public String toString() {
        StringBuilder result = new StringBuilder();
        StringBuilder logTrace = new StringBuilder();
        StringBuilder modelTrace = new StringBuilder();
        StringBuilder trace = new StringBuilder();
        result.append(String.format("\nTotal cost:%d\n", totalCost));
        for (Move m : moves) {
            result.append(m.toString()).append("\n");
            if (!m.getLogMove().equals(">>")) logTrace.append(m.getLogMove());

            if (!m.getModelMove().equals(">>")) modelTrace.append(m.getModelMove());
        }
        trace = logTrace;
        result.append("Trace: ").append(trace).append("\n");
        result.append("Log: ").append(logTrace).append("\n");
        result.append("Mod: ").append(modelTrace).append("\n");
        return result.toString();
    }

    public String toString(AlphabetService service) {
        StringBuilder result = new StringBuilder();
        StringBuilder logTrace = new StringBuilder();
        StringBuilder modelTrace = new StringBuilder();
        StringBuilder trace = new StringBuilder();
        result.append(String.format("Total cost:%d\n", totalCost));
        for (Move m : moves) {
            result.append(m.toString(service) + "\n");
            if (!m.getLogMove().equals(">>")) logTrace.append(service.deAlphabetize(m.getLogMove().charAt(0)));
            if (!m.getModelMove().equals(">>")) modelTrace.append(service.deAlphabetize(m.getModelMove().charAt(0)));
        }
        trace = logTrace;
        result.append("Trace: ").append(trace).append("\n");
        result.append("Log: ").append(logTrace).append("\n");
        result.append("Mod: ").append(modelTrace).append("\n");
        return result.toString();
    }

    public String toString(boolean compressed, int completenessCost) {
        StringBuilder result = new StringBuilder();
        //HashMap<String, Integer> encode = new HashMap<>();
        for(Move m : moves) {
            result.append(m.toString(true));
        }
        //result.append(totalCost);
        result.append(totalCost).append(completenessCost);
        return result.toString();
        /*for (Move m : moves) {
            String temp = m.toString(true);
            if (encode.containsKey(temp)) {
                encode.put(temp, encode.get(temp) + 1);
            } else {
                encode.put(temp, 1);
            }
            //result.append(m.toString(true));
        }
        encode.put("conf", totalCost);
        encode.put("comp", completenessCost);
        String mapAsString = encode.keySet().stream()
                .map(key -> encode.get(key).toString())
                .collect(Collectors.joining());
        result.append(mapAsString);
        return result.toString();*/
    }

    public List<Move> getMoves() {
        List<Move> result = new ArrayList<>();
        result.addAll(moves);
        return result;
    }

    public List<String> getPrefixTrace() {
        List<String> result = new ArrayList<>();
        for (Move m : moves) {
            if (!m.getLogMove().equals(">>")) {
                result.add(m.getLogMove());
            }
        }
        return result;
    }

    public int getTraceSize() {
        int result = 0;
        for (Move m : moves) {
            if (m.getLogMove().equals(">>")) {
                continue;
            } else {
                result++;
            }
        }
        return result;
    }


    public int getModelSize() {
        int result = 0;
        for (Move m : moves) {
            if (m.getModelMove().equals(">>")) {
                continue;
            } else {
                result++;
            }
        }
        return result;
    }

    public int hashCode() {
        return this.toString().hashCode();
    }

    public String logProjection() {
        StringBuilder sb = new StringBuilder();
        this.getMoves().stream().filter(x -> !x.getLogMove().equals(">>")).forEach(e -> sb.append(e.getLogMove().trim()));
        return sb.toString();
    }

    public String modelProjection() {
        StringBuilder sb = new StringBuilder();
        this.getMoves().stream().filter(x -> !x.getModelMove().equals(">>")).forEach(e -> sb.append(e));
        return sb.toString();
    }
}

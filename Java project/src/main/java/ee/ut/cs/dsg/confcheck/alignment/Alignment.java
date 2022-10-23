package ee.ut.cs.dsg.confcheck.alignment;

import ee.ut.cs.dsg.confcheck.util.AlphabetService;

import java.util.ArrayList;
import java.util.List;

public class Alignment {
    private List<Move> moves;

    public void setTotalCost(int totalCost) {
        this.totalCost = totalCost;
    }

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
            result.append(m.toString() + "\n");
            if (!m.getLogMove().equals(">>"))
                logTrace.append(m.getLogMove());
            if (!m.getModelMove().equals(">>"))
                modelTrace.append(m.getModelMove());

            if (!m.getLogMove().equals(">>")) {
                trace.append(m.getLogMove());
            } else if (!m.getModelMove().equals(">>")) {
                trace.append(m.getModelMove());
            }
        }
        result.append(("Trace: " + trace.toString() + "\n"));
        result.append("Log: " + logTrace.toString() + "\n");
        result.append("Mod: " + modelTrace.toString() + "\n");
        return result.toString();

    }

    public String toString(AlphabetService service) {
        StringBuilder result = new StringBuilder();
        StringBuilder logTrace = new StringBuilder();
        StringBuilder modelTrace = new StringBuilder();
        result.append(String.format("Total cost:%d\n", totalCost));
        for (Move m : moves) {
            result.append(m.toString(service) + "\n");
            if (!m.getLogMove().equals(">>"))
                logTrace.append(service.deAlphabetize(m.getLogMove().charAt(0)));
            if (!m.getModelMove().equals(">>"))
                modelTrace.append(service.deAlphabetize(m.getModelMove().charAt(0)));
        }
        result.append("Log: " + logTrace.toString() + "\n");
        result.append("Mod: " + modelTrace.toString());
        return result.toString();

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
            } else if (!m.getModelMove().equals(">>")) {
                result.add(m.getModelMove());
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

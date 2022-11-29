package ee.ut.cs.dsg.confcheck.alignment;

import ee.ut.cs.dsg.confcheck.util.AlphabetService;

import java.io.Serializable;

public class Move implements Serializable {
    private char logMove;
    private char modelMove;
    private int cost;
    private int oracle;

    public Move(String logMove, String modelMove, int cost)
    {
        this.cost = cost;
        this.logMove = logMove.charAt(0);
        this.modelMove = modelMove.charAt(0);
    }

    public int getCost() {
        return  cost;
    }

    public String toString()
    {
        return String.format("[logMove:%s, modelMove:%s, cost:%d]", logMove,modelMove,cost);
    }

    public String toString(boolean compressed)
    {
        StringBuilder stringBuilder = new StringBuilder();
        return stringBuilder.append(logMove+modelMove).append(cost).toString();
    }

    public String toString(AlphabetService service)
    {
        String l = service.deAlphabetize(logMove);
        String m = service.deAlphabetize(modelMove);
        return String.format("[logMove:%s, modelMove:%s, cost:%d]", l == null? ">>": l , m == null ? ">>":m,cost);
    }

    public String getModelMove(){return String.valueOf(modelMove);}
    public String getLogMove(){return  String.valueOf(logMove);}
}

package ee.ut.cs.dsg.confcheck.cost;

import ee.ut.cs.dsg.confcheck.ConformanceChecker;
import ee.ut.cs.dsg.confcheck.State;
import ee.ut.cs.dsg.confcheck.util.Configuration;

import java.util.List;

public class TripleCCostFunction implements CostFunction{

    @Override
    public int computeCost(State state, List<String> suffix, String event, Configuration.MoveType mt, ConformanceChecker conformanceChecker) {
        return 0;
    }
}

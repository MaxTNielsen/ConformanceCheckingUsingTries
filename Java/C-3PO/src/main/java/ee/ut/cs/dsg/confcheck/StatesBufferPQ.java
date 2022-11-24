package ee.ut.cs.dsg.confcheck;

import java.util.Comparator;
import java.util.PriorityQueue;

public class StatesBufferPQ {
    protected PriorityQueue<State> currentStates = new PriorityQueue<>(new theComparator());

    public StatesBufferPQ(String algString, State state) {
        currentStates.add(state);
    }

    public StatesBufferPQ(PriorityQueue<State> currentStates) {
        this.currentStates = currentStates;
    }

    public void setCurrentStates(PriorityQueue<State> currentStates) {
        this.currentStates = currentStates;
    }

    public PriorityQueue<State> getCurrentStates() {
        return currentStates;
    }

    static class theComparator implements Comparator<State> {
        @Override
        public int compare(State s1, State s2) {
            return s1.compareTo(s2);
        }
    }
}

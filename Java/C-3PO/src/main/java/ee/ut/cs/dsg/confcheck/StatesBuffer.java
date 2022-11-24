package ee.ut.cs.dsg.confcheck;

import ee.ut.cs.dsg.confcheck.alignment.Alignment;
import ee.ut.cs.dsg.confcheck.trie.TrieNode;

import java.io.Serializable;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

//

public class StatesBuffer implements Serializable {

    protected HashMap<String, State> currentStates;

    public StatesBuffer (String algString, State state){

        currentStates.put(algString, state);

    }

    public StatesBuffer (HashMap currentStates){

        this.currentStates = currentStates;

    }

    public void setCurrentStates(HashMap<String, State> currentStates){
        this.currentStates = currentStates;
    }

    public HashMap<String, State> getCurrentStates() {
        return currentStates;
    }

}

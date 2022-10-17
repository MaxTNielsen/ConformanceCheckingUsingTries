package ee.ut.cs.dsg.confcheck.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;

public class PredictionsClient {

    private final HashMap<String, String> urls;
    private HttpURLConnection con = null;

    public PredictionsClient(HashMap<String, String> urls) {
        this.urls = urls;
    }

    public int initModel(String urlKey) {
        try {
            con = createConnection(urlKey);
            con.setRequestMethod("GET");
            return con.getResponseCode();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public float getPrefixProb(String urlKey, String jsonInputString) {
        // String jsonInputString = "{"name": "Upendra", "job": "Programmer"}";
        int status;
        float retVal = 0;

        try {
            con = createConnection(urlKey);
            con.setRequestMethod("POST");
            con.setRequestProperty("Content-Type", "application/json");
            //con.setRequestProperty("Accept", "application/json");
            con.setDoOutput(true);

            try (OutputStream os = con.getOutputStream()) {
                byte[] input = jsonInputString.getBytes(StandardCharsets.UTF_8);
                os.write(input, 0, input.length);
            }

            status = con.getResponseCode();
            if (status == 200) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(con.getInputStream(), StandardCharsets.UTF_8))) {
                    StringBuilder response = new StringBuilder();
                    String responseLine;
                    while ((responseLine = br.readLine()) != null) {
                        response.append(responseLine.trim());
                    }
                    retVal = Float.parseFloat(response.toString());
                }
                return retVal;
            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (NullPointerException e) {
            System.out.println("connection is null");
        }
        return retVal;
    }

    private HttpURLConnection createConnection(String urlKey) {
        if (con != null)
            con.disconnect();

        try {
            URL url = new URL(urls.get(urlKey));
            con = (HttpURLConnection) url.openConnection();
        } catch (java.net.MalformedURLException e) {
            System.out.println(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return con;
    }

    public void closeConnection() {
        con.disconnect();
        con = null;
    }

}

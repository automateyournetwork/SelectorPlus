# NVD Database MCP Server
## Features
- Query specific CVEs by ID with detailed vulnerability data.
- Search the NVD database by keyword with customizable result options.
- Supports Server-Sent Events (SSE) transport for real-time communication.
- Compatible with MCP-compliant clients like Claude Desktop.

### Tools

The server implements the following tools to query the NVD Database:

- **`get_cve`**:
  - **Description**: Retrieves a CVE record by its ID.
  - **Parameters**:
    - `cve_id` (str): The CVE ID (e.g., `CVE-2019-1010218`).
    - `concise` (bool, default `False`): If `True`, returns a shorter format.
  - **Returns**: Detailed CVE info including scores, weaknesses, and references.

- **`search_cve`**:
  - **Description**: Searches the NVD database by keyword.
  - **Parameters**:
    - `keyword` (str): Search term (e.g., `Red Hat`).
    - `exact_match` (bool, default `False`): If `True`, requires an exact phrase match.
    - `concise` (bool, default `False`): If `True`, returns shorter CVE records.
    - `results` (int, default `10`): Maximum number of CVE records (1-2000).
  - **Returns**: List of matching CVEs with total count.

3. **Set Environment Variables**:
   - Create a `.env` file in the project root:
     ```
     NVD_API_KEY=your-api-key
     ```
   - Replace `your-api-key` with your NVD API key.
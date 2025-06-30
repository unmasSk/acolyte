# Example: Custom Dream Prompts

This example shows how to create custom analysis prompts for your project.

## Step 1: Create a prompts directory in your project

```bash
mkdir .acolyte/dream-prompts
```

## Step 2: Copy and customize the default prompts

```bash
cp src/acolyte/dream/prompts/*.md .acolyte/dream-prompts/
```

## Step 3: Configure in .acolyte

```yaml
dream:
  prompts_directory: ".acolyte/dream-prompts"
```

## Step 4: Or override specific prompts in config

```yaml
dream:
  prompts:
    bug_detection: |
      You are an expert in finding bugs in Python async code.
      Focus specifically on:
      - Asyncio race conditions
      - Missing await keywords
      - Improper exception handling in async contexts
      
      Analyze the code and report issues as JSON.
      
      Code to analyze:
      {code}
```

## Custom Prompt Best Practices

1. **Keep the JSON format**: Dream expects JSON responses for parsing
2. **Include {code} placeholder**: This is where the analyzed code is inserted
3. **Include {context} if using sliding window**: For preserved context between cycles
4. **Be specific**: Tailor prompts to your project's tech stack and concerns
5. **Test thoroughly**: Verify JSON parsing works with your custom prompts

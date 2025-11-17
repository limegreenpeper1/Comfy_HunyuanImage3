# Contributing to HunyuanImage-3.0 ComfyUI Nodes

Thank you for your interest in contributing!

**Project Maintainer**: Eric Hiss ([GitHub: EricRollei](https://github.com/ericRollei/))
**Contact**: eric@historic.camera, eric@rollei.us

We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- ComfyUI version, GPU model, VRAM amount
- Console logs if applicable

### Suggesting Features

Open an issue with:
- Clear description of the feature
- Use cases and benefits
- Proposed implementation (if you have ideas)

### Submitting Pull Requests

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**:
   - Follow existing code style
   - Add comments for complex logic
   - Update README.md if adding features
4. **Test thoroughly**:
   - Test with different GPU configs
   - Test with both NF4 and BF16 models
   - Verify no memory leaks
5. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: description"
   ```
6. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- Use descriptive variable names
- Add docstrings to functions/classes
- Keep functions focused and single-purpose
- Use type hints where helpful
- Follow PEP 8 style guide

## Testing Checklist

Before submitting:
- [ ] Code runs without errors
- [ ] No VRAM leaks (test with Unload node)
- [ ] Works with NF4 quantized model
- [ ] Works with full BF16 model (if applicable)
- [ ] Error messages are clear and helpful
- [ ] README updated (if adding features)

## Areas We Need Help With

- Testing on different GPU configurations
- Performance optimizations
- UI/UX improvements
- Documentation improvements
- Additional model variants support
- Example workflows

## Questions?

Open a discussion or issue - we're happy to help!

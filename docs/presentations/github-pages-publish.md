# GitHub Pages 发布说明

目标页面：

```text
docs/index.html
docs/presentations/x2-capx-leadership-report-20260702.html
```

推送后访问地址通常是：

```text
https://programessi.github.io/CaP-X-b1k/
```

本次页面需要提交这些文件：

```bash
git add .github/workflows/pages.yml \
  docs/.nojekyll \
  docs/index.html \
  docs/presentations/
git commit -m "Add X2 CaP-X GitHub Pages report"
git push origin main
```

如果 Pages 还没启用，在 GitHub 仓库页面进入：

```text
Settings -> Pages -> Build and deployment -> Source: GitHub Actions
```

然后在 Actions 页面查看 `Deploy GitHub Pages` workflow 是否成功。


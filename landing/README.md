# InboxBridge Landing

Public static landing experience for InboxBridge, implemented from the Stitch
screen `2e95e801d7ba476aa9e520ad7493ce28` in project
`15616762419389439509`.

The app uses real browser routes:

- `/`
- `/how-it-works`
- `/capabilities`
- `/safety`
- `/architecture`

## Local development

```bash
npm install
npm run dev
```

## Production build

```bash
npm run build
```

Deploy the generated `dist/` directory as a static site behind nginx or any
static hosting provider. Configure the server to fall back unknown paths to
`index.html` so direct navigation to a route continues to work. The frontend
does not require the InboxBridge runtime.

Inter and JetBrains Mono are bundled under `public/fonts/` to match the Stitch
design without a runtime Google Fonts dependency. The page uses CSS diagrams and
the downloaded Stitch preview only as a visual reference; no backend assets or
credentials are required.

The GitHub links point to the repository remote:
`https://github.com/darroyo083/inboxbridge`.

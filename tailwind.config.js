/** Build-time Tailwind config.
 *
 * Mirrors the inline `tailwind.config` that app/templates/base.html used to
 * hand the Play CDN, so the compiled stylesheet is a like-for-like
 * replacement. Pinned to the exact version the CDN was serving (3.4.17) to
 * avoid quietly changing utility behaviour at the same time as changing how
 * the CSS is delivered.
 */
module.exports = {
  darkMode: 'class',
  content: ['./app/templates/**/*.html'],

  // The scanner reads template SOURCE, so any class name assembled from a
  // Jinja variable is invisible to it. app/templates/_badge.html is the only
  // place that does this (`h-{{ size }} w-{{ size }}`); every other class in
  // the codebase — including the ones JavaScript toggles — appears as a
  // literal string in a template file and is picked up normally.
  //
  // Sizes below are every value passed to team_badge/competition_badge, plus
  // both macro defaults (8 and 6). Adding a new badge size means adding it
  // here, which tests/test_badge_safelist.py enforces.
  safelist: [
    'h-5', 'w-5',
    'h-6', 'w-6',
    'h-8', 'w-8',
    'h-10', 'w-10',
    'h-12', 'w-12',
    'h-14', 'w-14',
    'h-20', 'w-20',
  ],

  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
    },
  },
};

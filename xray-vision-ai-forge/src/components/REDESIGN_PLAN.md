# Component Redesign Plan - Clinical Clarity Theme

> Apple-inspired minimalism meets medical institutional trust

## Design System Reference

**Already completed:**
- `Landing.tsx` - Full redesign
- `Header.tsx` - Full redesign
- `Footer.tsx` - Full redesign
- `ChatSidebar.tsx` - Full redesign
- `SavedExperiments.tsx` - Full redesign
- `index.css` - Design tokens & utilities
- `tailwind.config.ts` - Medical color tokens

---

## Design Tokens Quick Reference

```
Primary:        hsl(172, 63%, 22%)   // Medical teal
Primary Dark:   hsl(172, 63%, 18%)
Primary Light:  hsl(172, 45%, 35%)
Mint:           hsl(168, 40%, 90%)
Sage:           hsl(165, 20%, 96%)
Trust Blue:     hsl(210, 100%, 96%)
Success:        hsl(152, 60%, 42%)
Warning:        hsl(35, 70%, 50%)
Error:          hsl(0, 72%, 51%)
Gray 50:        hsl(215, 15%, 55%)
Gray Dark:      hsl(172, 43%, 15%)
```

**Patterns:**
- Border radius: `rounded-xl` (cards), `rounded-2xl` (buttons/modals)
- Shadows: `shadow-md shadow-[hsl(172_63%_22%)]/15`
- Backgrounds: `bg-trust-gradient`, `bg-[hsl(168_25%_96%)]`
- Animations: `animate-fadeIn`, staggered delays
- Glass: `glass` class for frosted effects

---

## Components To Redesign

### Priority 1: Core Experiment Flow (High Impact)

#### 1. DatasetUpload.tsx (~940 lines)
**Current:** Standard card with blue/gray theme, generic styling
**Redesign scope:**
- [ ] Replace card with Apple-style rounded container
- [ ] Redesign drag-drop zone with teal gradient border on hover
- [ ] Update validation steps to use sage/mint backgrounds
- [ ] Style success state with teal checkmarks, not green
- [ ] Update slider to use teal track
- [ ] Replace blue info boxes with sage/mint variants
- [ ] Update buttons to use primary teal styling
- [ ] Add staggered fade-in animations to validation steps

#### 2. ExperimentConfig.tsx (~766 lines)
**Current:** Dense form with tabs, blue accents
**Redesign scope:**
- [ ] Redesign training mode selector (Centralized/Federated/Both) as Apple-style segmented control
- [ ] Update tabs to use underline style, not filled
- [ ] Style input fields with subtle sage backgrounds on focus
- [ ] Redesign parameter sections with clear visual hierarchy
- [ ] Update sliders to teal theme
- [ ] Style tooltips with glass morphism
- [ ] Add section dividers with gradient lines
- [ ] Update badges (Federated, Advanced) to theme colors

#### 3. TrainingExecution.tsx (~500 lines estimated)
**Current:** Progress bars, status indicators, log display
**Redesign scope:**
- [ ] Redesign progress bar with teal gradient fill
- [ ] Update status badges (Training, Validating, Complete)
- [ ] Style metrics cards with subtle borders and shadows
- [ ] Redesign log display with monospace font and sage background
- [ ] Add smooth progress animations
- [ ] Update epoch/round indicators with pill badges
- [ ] Style action buttons (Cancel, Pause) consistently

#### 4. ResultsVisualization.tsx (~1000 lines)
**Current:** Recharts with default styling, metric cards
**Redesign scope:**
- [ ] Update chart colors to theme palette (teal, mint, sage)
- [ ] Redesign metric cards with icon containers like SavedExperiments
- [ ] Style confusion matrix with theme colors
- [ ] Update tabs/sections with consistent styling
- [ ] Add subtle animations to metric reveals
- [ ] Style comparison views (Centralized vs Federated)
- [ ] Update tooltips with glass morphism

---

### Priority 2: Supporting Components (Medium Impact)

#### 5. StepIndicator.tsx (~75 lines)
**Current:** Circular steps with green/gray colors
**Redesign scope:**
- [ ] Replace green checkmarks with teal
- [ ] Update active step glow to teal
- [ ] Style connector lines with gradient
- [ ] Add smooth transition animations
- [ ] Update typography to Plus Jakarta Sans weights

#### 6. ProgressIndicator.tsx (~75 lines)
**Current:** Breadcrumb style with green accents
**Redesign scope:**
- [ ] Update active state to teal
- [ ] Replace green checkmarks with teal
- [ ] Style inactive steps with subtle gray
- [ ] Add pill-style background for current step

#### 7. InstructionCard.tsx (~100 lines estimated)
**Current:** Multiple variants (info, warning, success, tip, guide)
**Redesign scope:**
- [ ] Update info variant to use trust-blue/sage
- [ ] Update success variant to teal instead of green
- [ ] Update warning to amber/sage mix
- [ ] Update tip variant with mint background
- [ ] Update guide variant with subtle gradient
- [ ] Add subtle left border accent in theme colors
- [ ] Update icon colors to match variant themes

#### 8. HelpTooltip.tsx (~50 lines estimated)
**Current:** Standard tooltip with icon
**Redesign scope:**
- [ ] Apply glass morphism to tooltip popup
- [ ] Update icon to theme gray
- [ ] Add subtle shadow to tooltip
- [ ] Update typography weights

---

### Priority 3: Utility Components (Lower Impact)

#### 9. WelcomeGuide.tsx (~274 lines)
**Current:** Multi-step modal dialog with blue accents
**Redesign scope:**
- [ ] Apply glass morphism to dialog
- [ ] Update step indicators to teal
- [ ] Style content cards with sage backgrounds
- [ ] Update buttons to primary teal
- [ ] Replace emoji with refined icons
- [ ] Add subtle slide animations between steps

#### 10. LoadingOverlay.tsx (~49 lines)
**Current:** Simple loader with medical class
**Redesign scope:**
- [ ] Update overlay background with subtle blur
- [ ] Style loader card with rounded-2xl
- [ ] Add subtle shadow
- [ ] Update spinner color to teal

#### 11. MetadataDisplay.tsx (~336 lines)
**Current:** Card-based with gradient headers
**Redesign scope:**
- [ ] Update gradient headers to teal theme
- [ ] Style badges with theme colors
- [ ] Update icon colors to muted teal
- [ ] Refine typography hierarchy
- [ ] Update nested object borders

#### 12. StepContent.tsx (~48 lines)
**Current:** Simple fade animation wrapper
**Redesign scope:**
- [ ] Verify animations work with new theme
- [ ] No major changes expected

---

## Implementation Order

**Session 1:** DatasetUpload.tsx (highest user interaction)
**Session 2:** ExperimentConfig.tsx (complex forms)
**Session 3:** TrainingExecution.tsx (progress/status)
**Session 4:** ResultsVisualization.tsx (charts/metrics)
**Session 5:** StepIndicator.tsx + ProgressIndicator.tsx
**Session 6:** InstructionCard.tsx + HelpTooltip.tsx
**Session 7:** WelcomeGuide.tsx + LoadingOverlay.tsx + MetadataDisplay.tsx

---

## Shadcn/UI Components

The `/ui` folder contains 47 Shadcn components. These should NOT be directly modified. Instead:

1. Override styles via `index.css` using CSS variables
2. Use className prop to apply theme styles
3. Create wrapper components if needed for consistent theming

Key Shadcn components to style via usage:
- `Button` - Apply `bg-[hsl(172_63%_22%)]` classes
- `Card` - Apply `rounded-2xl` and theme shadows
- `Dialog` - Apply glass morphism via className
- `Slider` - Style track/thumb via CSS
- `Tabs` - Style via className for underline variant
- `Badge` - Create theme variants via className

---

## CSS Utilities to Add (if needed)

```css
/* Add to index.css as needed */

.input-focus-sage {
  @apply focus:bg-[hsl(168_25%_98%)] focus:border-[hsl(172_63%_35%)];
}

.slider-teal {
  @apply [&_[role=slider]]:bg-[hsl(172_63%_22%)];
}

.tab-underline {
  @apply border-b-2 border-transparent data-[state=active]:border-[hsl(172_63%_22%)];
}
```

---

## Testing Checklist (Per Component)

- [ ] Visual consistency with completed components
- [ ] Dark/light text contrast passes WCAG AA
- [ ] Animations are smooth (60fps)
- [ ] Responsive at mobile/tablet/desktop
- [ ] All interactive states styled (hover, focus, active, disabled)
- [ ] Loading states match theme
- [ ] Error states use consistent red
- [ ] Success states use teal, not green

---

## Notes

- Keep functional code unchanged - only modify styling
- Preserve all TypeScript interfaces and props
- Test each component in isolation before integration
- Take screenshots before/after for comparison
- Commit after each component redesign

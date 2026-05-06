"""add version_id to active_models

Revision ID: a1b2c3d4e5f6
Revises: 9ea8eb380bbe
Create Date: 2026-04-29 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '6e66f3ee876a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add version_id column to active_models."""
    op.add_column(
        'active_models',
        sa.Column(
            'version_id',
            sa.String(length=64),
            nullable=True,
            comment='Bundle version ID (YYYYMMDD_HHMM_modeltype)',
        ),
    )


def downgrade() -> None:
    """Remove version_id column from active_models."""
    op.drop_column('active_models', 'version_id')
